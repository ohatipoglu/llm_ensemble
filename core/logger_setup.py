# core/logger_setup.py

import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger():
    """
    Kök logger'ı yapılandırır:
    - 10MB limitli, 5 yedekli UTF-8 RotatingFileHandler (logs/app.log)
    - Konsol çıktısı için StreamHandler
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, "app.log")
    
    # Kök (root) logger yapılandırması
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Mevcut tüm handler'ları temizle
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Log formatı
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. Dosyaya Yazma (RotatingFileHandler - 10MB limitli, 5 yedekli)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 2. Konsola Yazma (StreamHandler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logger.info(f"Sistem: Loglama altyapısı başarıyla kuruldu. Dosya: {log_file}")
    
    # Kök logger'ı döndür
    return logger
