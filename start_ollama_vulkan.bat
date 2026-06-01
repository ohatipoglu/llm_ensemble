@echo off
setlocal

@REM Vulkan donanimsal hizlandirmasini zorunlu kil! (Intel Arc iGPU icin)
set OLLAMA_VULKAN=1

@REM Modelleri RAM'de surekli tut
set OLLAMA_KEEP_ALIVE=-1

@REM Ayni anda Continue.dev icin 2 model calistirma izni
set OLLAMA_NUM_PARALLEL=2
set OLLAMA_HOST=127.0.0.1:11434

echo Standart Ollama, Vulkan (Intel Arc) destegiyle baslatiliyor...
ollama serve