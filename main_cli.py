import asyncio
import pandas as pd
# Import'lar artık çok daha basit. Doğrudan klasör adlarından başlıyor.
from api.api_manager import APIManager
from db.database_manager import DatabaseManager
from core.conversation_manager import ConversationManager

pd.set_option('display.max_colwidth', None)

async def main():
    # --- YEREL MODELLERİNİZİ ROLLERLE EŞLEŞTİRİN ---
    # Model adlarını Ollama'da yüklü olan tam isimleriyle yazın.
    MODELS = {
        "responder1": "qwen3.5:9b",
        "responder2": "gemma4:26B-32K",
        "critic": "qwen3.5:9b",
        "judge": "qwen3.6:35b"
    }

    try:
        api_manager = APIManager()
        db_manager = DatabaseManager()
        conversation_manager = ConversationManager(api_manager, db_manager)

        user_prompt = input("Lütfen LLM'lerin tartışmasını istediğiniz konuyu veya soruyu girin:\n> ")

        if not user_prompt:
            print("Geçerli bir soru girmediniz. Program sonlandırılıyor.")
            return

        seq_input = input("Ardışıl (Sequential) mod aktif edilsin mi? (E/H - Varsayılan: H): ").strip().lower()
        api_manager.sequential_mode = seq_input in ('e', 'y', 'yes', 'true', '1')

        # Konuşmayı veritabanında oluşturup ID'sini alıyoruz
        conversation_id = db_manager.create_conversation(user_prompt, MODELS)
        results = await conversation_manager.run_full_conversation(user_prompt, MODELS, conversation_id)
        
        print("\n" + "#"*60)
        print(" YEREL TARTIŞMA SONUCU ".center(60, "#"))
        print("#"*60 + "\n")
        
        print(f"SORU: {results['prompt']}\n")
        
        df_data = {
            "ROL": [
                f"YANIT VERİCİ 1 ({results['responder1']['model']})", 
                " ",
                f"YANIT VERİCİ 2 ({results['responder2']['model']})",
                " ",
                f"ELEŞTİRMEN ({results['critic']['model']})",
                " ",
                f"YÜKSEK HAKİM ({results['judge']['model']})"
            ],
            "AŞAMA": ["İlk Yanıt", "Savunma", "İlk Yanıt", "Savunma", "Eleştiri", "Son Söz", "Nihai Karar"],
            "İÇERİK": [
                results['responder1']['response'],
                results['responder1']['defense'],
                results['responder2']['response'],
                results['responder2']['defense'],
                results['critic']['critique'],
                results['critic']['final_word'],
                results['judge']['verdict']
            ]
        }
        df = pd.DataFrame(df_data)
        print(df.to_string())

    except Exception as e:
        print(f"\nAna programda bir hata oluştu: {e}")
    finally:
        if 'db_manager' in locals():
            db_manager.close()
            print("\nProgram bitti.")

if __name__ == "__main__":
    from core.logger_setup import setup_logger
    logger = setup_logger()
    logger.info("CLI Uygulaması başlatılıyor...")
    asyncio.run(main())