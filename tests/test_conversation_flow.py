import pytest
import asyncio
from unittest.mock import MagicMock
from core.conversation_manager import ConversationManager, Stage
from db.database_manager import DatabaseManager
from api.api_manager import APIManager

@pytest.mark.asyncio
async def test_full_debate_flow(tmp_path):
    # Test veritabanı dosyası için izole bir temp dizini kullan
    db_file = tmp_path / "test_llm_challenger.db"
    db_manager = DatabaseManager(db_path=str(db_file))
    
    # API Manager'ı mock'la
    api_manager = MagicMock(spec=APIManager)
    api_manager.host = "http://localhost:11434"
    api_manager.sequential_mode = False
    
    # call_llm_stream metotunun mock chunk'lar döndürmesini sağla
    async def mock_stream(*args, **kwargs):
        yield "Test "
        yield "yanıtı"
        
    api_manager.call_llm_stream = MagicMock(side_effect=mock_stream)
    
    conversation_manager = ConversationManager(api_manager, db_manager)
    
    MODELS = {
        "responder1": "mock-model-1",
        "responder2": "mock-model-2",
        "critic": "mock-critic",
        "judge": "mock-judge"
    }
    
    user_prompt = "Yapay zeka gelecekte işlerimizi elimizden alacak mı?"
    
    # 1. Veritabanında konuşmayı oluştur
    conv_id = db_manager.create_conversation(user_prompt, MODELS)
    assert conv_id > 0
    
    # 2. Tartışma akışını baştan sona simüle et
    results = await conversation_manager.run_full_conversation(
        user_prompt=user_prompt,
        models=MODELS,
        conversation_id=conv_id
    )
    
    # 3. Sonuçların doğruluğunu test et
    assert results is not None
    assert results["prompt"] == user_prompt
    assert results["responder1"]["response"] == "Test yanıtı"
    assert results["responder2"]["response"] == "Test yanıtı"
    assert results["critic"]["critique"] == "Test yanıtı"
    assert results["judge"]["verdict"] == "Test yanıtı"
    
    # 4. Konuşmanın tamamlandığını doğrula (get_incomplete_conversation None dönmeli)
    incomplete = db_manager.get_incomplete_conversation()
    assert incomplete is None
    
    # Veritabanını güvenle kapat
    db_manager.close()
