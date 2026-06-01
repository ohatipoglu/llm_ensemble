# core/conversation_manager.py

import asyncio
import logging
import os
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from api.api_manager import APIManager
from db.database_manager import DatabaseManager

logger = logging.getLogger("conversation_manager")

def strip_cot_thinking(response: str) -> str:
    """
    Model yanıtındaki <thinking>...</thinking> bloklarını temizler
    ve sadece final metnini döndürür.
    """
    if not response:
        return ""
    # <thinking>...</thinking> bloğunu regex ile temizle (re.DOTALL ile satır sonlarını da kapsar)
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', response, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

class Stage(Enum):
    STARTED = 0
    RESPONDERS_DONE = 1
    CRITIC_DONE = 2
    DEFENSES_DONE = 3
    FINAL_WORD_DONE = 4
    COMPLETED = 5

@dataclass
class ConversationState:
    stage: Stage
    responder1_response: str = ""
    responder2_response: str = ""
    critic_response: str = ""
    defense1_response: str = ""
    defense2_response: str = ""
    final_word_response: str = ""
    judge_response: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "responder1": self.responder1_response,
            "responder2": self.responder2_response,
            "critic": self.critic_response,
            "defense1": self.defense1_response,
            "defense2": self.defense2_response,
            "final_word": self.final_word_response,
            "judge": self.judge_response
        }

class ConversationManager:
    def __init__(self, api_manager: APIManager, db_manager: DatabaseManager):
        self.api_manager = api_manager
        self.db_manager = db_manager
        self._cancelled = False
        self._pause_requested = False
        self._conversation_id: Optional[int] = None
        self._models: Optional[Dict] = None
        self._user_prompt: Optional[str] = None

    def cancel(self):
        self._cancelled = True

    def request_pause(self):
        """Mevcut adım bitince durdurmayı talep eder"""
        self._pause_requested = True
        logger.info("Duraklatma talebi alındı (mevcut adım bitince duracak)")

    def _check_pause(self, current_stage: Stage, state_data: Dict):
        """Her adım sonunda duraklatma kontrolü"""
        if self._pause_requested and self._conversation_id:
            logger.info(f"Aşama {current_stage.name} tamamlandı, duraklatılıyor...")
            self.db_manager.mark_conversation_paused(
                self._conversation_id, current_stage.value, state_data
            )
            return True
        return False

    async def run_full_conversation(
        self, 
        user_prompt: str, 
        models: Dict[str, str],
        conversation_id: int,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        chunk_callback: Optional[Callable[[str, str], None]] = None,  # (model_key, chunk)
        start_stage: Stage = Stage.STARTED,
        resume_state: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Çok aşamalı tartışma akışı.
        chunk_callback: Her gelen parça için çağrılır (real-time update)
        """
        self._user_prompt = user_prompt
        self._models = models
        self._conversation_id = conversation_id
        self._cancelled = False
        self._pause_requested = False

        # OpenTelemetry Context & Root Span Başlat
        from core.telemetry import tracer, get_conversation_context
        ctx = get_conversation_context(conversation_id)
        
        span_ctx = None
        span = None
        try:
            span_ctx = tracer.start_as_current_span("run_full_conversation", context=ctx)
            if span_ctx:
                span = span_ctx.__enter__()
                if span:
                    span.set_attribute("conversation.id", conversation_id)
                    span.set_attribute("stage.start", start_stage.name)
        except Exception as te:
            logger.warning(f"Telemetry error starting conversation span: {te}")
            span_ctx = None

        logger.info(f"Konuşma Başlatılıyor (ID: {conversation_id}), Başlangıç Aşaması: {start_stage.name}")

        try:
            # State'leri tut
            state = ConversationState(stage=start_stage)
            if resume_state:
                state.responder1_response = resume_state.get('responder1', '')
                state.responder2_response = resume_state.get('responder2', '')
                state.critic_response = resume_state.get('critic', '')
                state.defense1_response = resume_state.get('defense1', '')
                state.defense2_response = resume_state.get('defense2', '')
                state.final_word_response = resume_state.get('final_word', '')
                state.judge_response = resume_state.get('judge', '')

            # AŞAMA 1: Yanıt Vericiler (Eğer başlangıç aşaması 0 ise)
            if start_stage.value <= Stage.STARTED.value:
                if progress_callback:
                    progress_callback(10, "Responder 1 ve 2'ye soru gönderiliyor...")
                
                # Prompt'ları kaydet (sadece ilk sefer)
                if not resume_state:
                    self.db_manager.save_prompt(conversation_id, models['responder1'], 'yanıt', user_prompt)
                    self.db_manager.save_prompt(conversation_id, models['responder2'], 'yanıt', user_prompt)

                # Parallel streaming (OTel enstrümantasyonlu yardımcı fonksiyonlarla)
                task1 = self._run_responder(models['responder1'], user_prompt, 'responder1', chunk_callback)
                task2 = self._run_responder(models['responder2'], user_prompt, 'responder2', chunk_callback)
                
                state.responder1_response, state.responder2_response = await asyncio.gather(task1, task2)
                
                if self._cancelled:
                    return None
                
                # Yanıtları kaydet
                self.db_manager.save_response(conversation_id, models['responder1'], 'yanıt', state.responder1_response)
                self.db_manager.save_response(conversation_id, models['responder2'], 'yanıt', state.responder2_response)
                state.stage = Stage.RESPONDERS_DONE
                
                if self._check_pause(state.stage, state.to_dict()):
                    return None

            # AŞAMA 2: Eleştirmen
            if start_stage.value <= Stage.RESPONDERS_DONE.value:
                if progress_callback:
                    progress_callback(30, "Eleştirmen analiz yapıyor...")
                
                critic_prompt = self._build_critic_prompt(user_prompt, models, state.responder1_response, state.responder2_response)
                self.db_manager.save_prompt(conversation_id, models['critic'], 'eleştiri', critic_prompt)
                
                state.critic_response = await self._run_critic(
                    models['critic'], critic_prompt, 'critic', chunk_callback
                )
                
                if self._cancelled:
                    return None
                
                self.db_manager.save_response(conversation_id, models['critic'], 'eleştiri', state.critic_response)
                state.stage = Stage.CRITIC_DONE
                
                if self._check_pause(state.stage, state.to_dict()):
                    return None

            # AŞAMA 3: Savunmalar
            if start_stage.value <= Stage.CRITIC_DONE.value:
                if progress_callback:
                    progress_callback(50, "Responder'lar savunma hazırlıyor...")
                
                d1_prompt = self._build_defense_prompt(user_prompt, state.responder1_response, state.critic_response)
                d2_prompt = self._build_defense_prompt(user_prompt, state.responder2_response, state.critic_response)
                
                self.db_manager.save_prompt(conversation_id, models['responder1'], 'savunma', d1_prompt)
                self.db_manager.save_prompt(conversation_id, models['responder2'], 'savunma', d2_prompt)

                task1 = self._run_responder(models['responder1'], d1_prompt, 'defense1', chunk_callback)
                task2 = self._run_responder(models['responder2'], d2_prompt, 'defense2', chunk_callback)
                
                state.defense1_response, state.defense2_response = await asyncio.gather(task1, task2)
                
                if self._cancelled:
                    return None
                
                self.db_manager.save_response(conversation_id, models['responder1'], 'savunma', state.defense1_response)
                self.db_manager.save_response(conversation_id, models['responder2'], 'savunma', state.defense2_response)
                state.stage = Stage.DEFENSES_DONE
                
                if self._check_pause(state.stage, state.to_dict()):
                    return None

            # AŞAMA 4: Eleştirmen Son Söz
            if start_stage.value <= Stage.DEFENSES_DONE.value:
                if progress_callback:
                    progress_callback(70, "Eleştirmen son sözünü söylüyor...")
                
                final_prompt = self._build_final_word_prompt(
                    user_prompt, models, state.responder1_response, state.responder2_response,
                    state.critic_response, state.defense1_response, state.defense2_response
                )
                self.db_manager.save_prompt(conversation_id, models['critic'], 'son_söz', final_prompt)
                
                state.final_word_response = await self._run_critic(
                    models['critic'], final_prompt, 'final_word', chunk_callback
                )
                
                if self._cancelled:
                    return None
                
                self.db_manager.save_response(conversation_id, models['critic'], 'son_söz', state.final_word_response)
                state.stage = Stage.FINAL_WORD_DONE
                
                if self._check_pause(state.stage, state.to_dict()):
                    return None

            # AŞAMA 5: Yüksek Hakim
            if start_stage.value <= Stage.FINAL_WORD_DONE.value:
                if progress_callback:
                    progress_callback(90, "Yüksek Hakim kararını veriyor...")
                
                judge_prompt = self._build_judge_prompt(
                    user_prompt, models, state.responder1_response, state.responder2_response,
                    state.critic_response, state.defense1_response, state.defense2_response, 
                    state.final_word_response
                )
                self.db_manager.save_prompt(conversation_id, models['judge'], 'karar', judge_prompt)
                
                state.judge_response = await self._run_judge(
                    models['judge'], judge_prompt, chunk_callback
                )
                
                if self._cancelled:
                    return None
                
                self.db_manager.save_response(conversation_id, models['judge'], 'karar', state.judge_response)
                state.stage = Stage.COMPLETED
 
            if progress_callback:
                progress_callback(100, "Tamamlandı!")

            # Tamamlandı olarak işaretle
            self.db_manager.mark_conversation_completed(conversation_id)

            return {
                "prompt": user_prompt,
                "responder1": {"model": models['responder1'], "response": state.responder1_response, "defense": state.defense1_response},
                "responder2": {"model": models['responder2'], "response": state.responder2_response, "defense": state.defense2_response},
                "critic": {"model": models['critic'], "critique": state.critic_response, "final_word": state.final_word_response},
                "judge": {"model": models['judge'], "verdict": state.judge_response}
            }

        except asyncio.CancelledError:
            logger.info("Konuşma iptal edildi.")
            return None
        except Exception as e:
            logger.error(f"Konuşma hatası: {e}")
            raise
        finally:
            if span_ctx:
                try:
                    span_ctx.__exit__(None, None, None)
                except Exception as te:
                    logger.warning(f"Telemetry error exiting conversation span: {te}")

    async def _stream_llm(self, model: str, prompt_or_messages: Any, model_key: str, 
                         chunk_callback: Optional[Callable[[str, str], None]]) -> str:
        """LLM'i çağırır ve chunk'ları callback ile gönderir. Tüm metni döndürür."""
        if isinstance(prompt_or_messages, list):
            messages = prompt_or_messages
        else:
            messages = [{"role": "user", "content": prompt_or_messages}]
            
        full_content = ""
        try:
            async for chunk in self.api_manager.call_llm_stream(model, messages):
                if self._cancelled:
                    raise asyncio.CancelledError()
                
                full_content += chunk
                if chunk_callback:
                    chunk_callback(model_key, chunk)  # Real-time GUI update
        except Exception as e:
            logger.error(f"Model {model} çağrısında hata oluştu: {e}")
            raise
            
        # Boş yanıt kontrolü ve fallback mekanizması
        if not full_content.strip():
            fallback_msg = f"[Sistem Bildirimi: {model} modelinden geçerli bir yanıt alınamadı.]"
            logger.warning(f"Model {model} boş yanıt döndü. Fallback mesajı kullanılıyor.")
            if chunk_callback:
                chunk_callback(model_key, fallback_msg)
            return fallback_msg
            
        return strip_cot_thinking(full_content)

    async def _run_responder(self, model: str, prompt: Any, role: str, chunk_callback: Optional[Callable]) -> str:
        """Responder çağrısını tracer span ile sarmalar."""
        from core.telemetry import tracer
        import opentelemetry.trace as otel_trace

        prompt_str = ""
        if isinstance(prompt, list):
            prompt_str = " ".join(m.get('content', '') for m in prompt)
        else:
            prompt_str = str(prompt)
        tokens_approx = len(prompt_str.split()) if prompt_str else 0
        
        span_ctx = None
        span = None
        try:
            span_ctx = tracer.start_as_current_span("run_responder")
            if span_ctx:
                span = span_ctx.__enter__()
                if span:
                    span.set_attribute("agent.role", role)
                    span.set_attribute("stage.name", "responder")
                    span.set_attribute("input.tokens_approx", tokens_approx)
        except Exception as te:
            logger.warning(f"Telemetry error starting responder span: {te}")
            span_ctx = None

        status = "success"
        try:
            return await self._stream_llm(model, prompt, role, chunk_callback)
        except Exception as e:
            status = "error"
            if span:
                try:
                    span.record_exception(e)
                    span.set_status(otel_trace.StatusCode.ERROR, str(e))
                except Exception as te:
                    logger.warning(f"Telemetry error recording exception in responder: {te}")
            raise
        finally:
            if span_ctx:
                try:
                    if span:
                        span.set_attribute("output.status", status)
                    span_ctx.__exit__(None, None, None)
                except Exception as te:
                    logger.warning(f"Telemetry error exiting responder span: {te}")

    async def _run_critic(self, model: str, prompt: Any, role: str, chunk_callback: Optional[Callable]) -> str:
        """Critic/Final Word çağrısını tracer span ile sarmalar."""
        from core.telemetry import tracer
        import opentelemetry.trace as otel_trace

        prompt_str = ""
        if isinstance(prompt, list):
            prompt_str = " ".join(m.get('content', '') for m in prompt)
        else:
            prompt_str = str(prompt)
        tokens_approx = len(prompt_str.split()) if prompt_str else 0
        
        span_ctx = None
        span = None
        try:
            span_ctx = tracer.start_as_current_span("run_critic")
            if span_ctx:
                span = span_ctx.__enter__()
                if span:
                    span.set_attribute("agent.role", "critic")
                    span.set_attribute("stage.name", role)
                    span.set_attribute("input.tokens_approx", tokens_approx)
        except Exception as te:
            logger.warning(f"Telemetry error starting critic span: {te}")
            span_ctx = None

        status = "success"
        try:
            return await self._stream_llm(model, prompt, role, chunk_callback)
        except Exception as e:
            status = "error"
            if span:
                try:
                    span.record_exception(e)
                    span.set_status(otel_trace.StatusCode.ERROR, str(e))
                except Exception as te:
                    logger.warning(f"Telemetry error recording exception in critic: {te}")
            raise
        finally:
            if span_ctx:
                try:
                    if span:
                        span.set_attribute("output.status", status)
                    span_ctx.__exit__(None, None, None)
                except Exception as te:
                    logger.warning(f"Telemetry error exiting critic span: {te}")

    async def _run_judge(self, model: str, prompt: Any, chunk_callback: Optional[Callable]) -> str:
        """Judge çağrısını tracer span ile sarmalar."""
        from core.telemetry import tracer
        import opentelemetry.trace as otel_trace

        prompt_str = ""
        if isinstance(prompt, list):
            prompt_str = " ".join(m.get('content', '') for m in prompt)
        else:
            prompt_str = str(prompt)
        tokens_approx = len(prompt_str.split()) if prompt_str else 0
        
        span_ctx = None
        span = None
        try:
            span_ctx = tracer.start_as_current_span("run_judge")
            if span_ctx:
                span = span_ctx.__enter__()
                if span:
                    span.set_attribute("agent.role", "judge")
                    span.set_attribute("stage.name", "judge")
                    span.set_attribute("input.tokens_approx", tokens_approx)
        except Exception as te:
            logger.warning(f"Telemetry error starting judge span: {te}")
            span_ctx = None

        status = "success"
        try:
            return await self._stream_llm(model, prompt, 'judge', chunk_callback)
        except Exception as e:
            status = "error"
            if span:
                try:
                    span.record_exception(e)
                    span.set_status(otel_trace.StatusCode.ERROR, str(e))
                except Exception as te:
                    logger.warning(f"Telemetry error recording exception in judge: {te}")
            raise
        finally:
            if span_ctx:
                try:
                    if span:
                        span.set_attribute("output.status", status)
                    span_ctx.__exit__(None, None, None)
                except Exception as te:
                    logger.warning(f"Telemetry error exiting judge span: {te}")

    def _build_critic_prompt(self, user_prompt: str, models: Dict, r1: str, r2: str) -> List[Dict[str, str]]:
        system_content = "Sen iki yapay zeka modelinin yanıtlarını tarafsız ve eleştirel bir gözle analiz eden, güçlü ve zayıf yönlerini belirten bir Eleştirmen ajansın. Orijinal soruya verilen yanıtları detaylıca değerlendir."
        if os.getenv("ENABLE_COT", "false").lower() == "true":
            system_content += " Kararını vermeden önce gerekçelerini adım analiz et. Düşünce sürecini <thinking>...</thinking> blokları arasına yaz. Bloktan sonra SADECE nihai kararı/özetini ver. Gereksiz uzatma yapma."

        return [
            {
                "role": "system", 
                "content": system_content
            },
            {
                "role": "user", 
                "content": f"ORİJİNAL SORU:\n{user_prompt}\n\nYANIT 1 ({models['responder1']}):\n{r1}\n\nYANIT 2 ({models['responder2']}):\n{r2}"
            }
        ]

    def _build_defense_prompt(self, user_prompt: str, response: str, critique: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system", 
                "content": "Sen ilk yanıtını eleştiriye karşı savunan bir modelsin. İlk yanıtının neden doğru veya mantıklı olduğunu açıkla, eleştirideki haklı yönleri kabul et ama kendi pozisyonunu savun."
            },
            {
                "role": "user", 
                "content": f"ORİJİNAL SORU:\n{user_prompt}\n\nSENİN İLK YANITIN:\n{response}\n\nGELEN ELEŞTİRİ:\n{critique}"
            }
        ]

    def _build_final_word_prompt(self, user_prompt: str, models: Dict, r1: str, r2: str, 
                                  crit: str, d1: str, d2: str) -> List[Dict[str, str]]:
        return [
            {
                "role": "system", 
                "content": "Sen eleştirmen rolündesin. İlk yaptığın eleştirileri ve yanıt vericilerin kendilerini savundukları metinleri inceleyerek son sözünü (nihai analizini) söyle."
            },
            {
                "role": "user", 
                "content": f"ORİJİNAL SORU:\n{user_prompt}\n\nYanıt 1 ({models['responder1']}):\n{r1}\n\nYanıt 2 ({models['responder2']}):\n{r2}\n\nİlk Eleştirin:\n{crit}\n\nSavunma 1:\n{d1}\n\nSavunma 2:\n{d2}"
            }
        ]

    def _build_judge_prompt(self, user_prompt: str, models: Dict, r1: str, r2: str, 
                            crit: str, d1: str, d2: str, final: str) -> List[Dict[str, str]]:
        system_content = "Sen Yüksek Hakimsin. Önündeki tüm tartışmayı (soru, ilk yanıtlar, eleştiriler, savunmalar ve eleştirmenin son sözü) değerlendirerek nihai kararını ve gerekçeli karar sentezini ver."
        if os.getenv("ENABLE_COT", "false").lower() == "true":
            system_content += " Kararını vermeden önce gerekçelerini adım analiz et. Düşünce sürecini <thinking>...</thinking> blokları arasına yaz. Bloktan sonra SADECE nihai kararı/özetini ver. Gereksiz uzatma yapma."

        return [
            {
                "role": "system", 
                "content": system_content
            },
            {
                "role": "user", 
                "content": f"SORU:\n{user_prompt}\n\nYANIT 1 ({models['responder1']}):\n- İlk: {r1}\n- Savunma: {d1}\n\nYANIT 2 ({models['responder2']}):\n- İlk: {r2}\n- Savunma: {d2}\n\nELEŞTİRMEN ({models['critic']}):\n- Eleştiri: {crit}\n- Son Söz: {final}"
            }
        ]