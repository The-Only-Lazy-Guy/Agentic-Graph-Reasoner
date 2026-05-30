@echo off
REM ============================================================================
REM  setup_datagen_env.bat  -  fresh-environment setup for V4 data generation
REM
REM  Stands up everything needed to run answerer_v4 / run_phase15_corpus.py and
REM  produce traces + scoped_patches for the V5 corpus, as if on a brand-new box.
REM
REM  Steps: python venv -> runtime deps -> LLM backend (opencode CLI or local
REM  llama-server) -> env vars -> prints the exact generate command.
REM
REM  Usage:
REM    setup_datagen_env.bat                 :: full setup, no generation
REM    setup_datagen_env.bat --run           :: setup + kick off a generation run
REM    setup_datagen_env.bat --gpu           :: install CUDA torch (default: CPU)
REM
REM  Backend (pick one, configured AFTER this script):
REM    A) opencode CLI  : npm install -g opencode-ai ; opencode auth login
REM    B) llama-server  : serve a GGUF on http://127.0.0.1:6768  (LOCAL_LLM_BASE_URL)
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv-datagen"
set "DATASET=artifacts\phase15_test_20.json"
set "GRAPH=graphs\merged_graph.json"
set "OUT_DIR=artifacts\phase15"
set "CORPUS_FILE=phase15_corpus.jsonl"
set "DO_RUN=0"
set "GPU=0"
for %%A in (%*) do (
    if /I "%%A"=="--run" set "DO_RUN=1"
    if /I "%%A"=="--gpu" set "GPU=1"
)

echo ==========================================================
echo [1/6] Python venv: %VENV%
echo ==========================================================
where python >nul 2>nul || (echo ERROR: python not found on PATH. Install Python 3.10-3.12. & exit /b 1)
if not exist "%VENV%\Scripts\python.exe" (
    python -m venv "%VENV%" || (echo ERROR: venv creation failed. & exit /b 1)
)
call "%VENV%\Scripts\activate.bat"
python -m pip install --upgrade pip setuptools wheel

echo.
echo ==========================================================
echo [2/6] PyTorch ( %GPU%==1 means CUDA )
echo ==========================================================
if "%GPU%"=="1" (
    pip install torch --index-url https://download.pytorch.org/whl/cu124
) else (
    pip install torch --index-url https://download.pytorch.org/whl/cpu
)

echo.
echo ==========================================================
echo [3/6] Runtime deps (requirements-datagen.txt)
echo ==========================================================
pip install -r requirements-datagen.txt || (echo ERROR: pip install failed. & exit /b 1)

echo.
echo ==========================================================
echo [4/6] Warm the embedder cache (all-MiniLM-L6-v2)
echo ==========================================================
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('embedder cached')" || echo WARN: embedder warm failed (will download on first run).

echo.
echo ==========================================================
echo [5/6] LLM backend
echo ==========================================================
where opencode >nul 2>nul
if %errorlevel%==0 (
    echo   opencode found on PATH.
) else (
    where npm >nul 2>nul
    if !errorlevel!==0 (
        echo   installing opencode-ai globally via npm...
        call npm install -g opencode-ai || echo WARN: opencode install failed - use the llama-server backend instead.
    ) else (
        echo   Node/npm not found. For the opencode backend install Node 18+ then: npm install -g opencode-ai
        echo   OR use the local llama-server backend (Option B below).
    )
)

echo.
echo ==========================================================
echo [6/6] Environment variables (this session)
echo ==========================================================
set "LOCAL_LLM_BASE_URL=http://127.0.0.1:6768"
set "LOCAL_LLM_MAX_TOKENS=2400"
set "LOCAL_LLM_TEMPERATURE=0.2"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "PYTHONUNBUFFERED=1"
echo   LOCAL_LLM_BASE_URL=%LOCAL_LLM_BASE_URL%
echo   KMP_DUPLICATE_LIB_OK=%KMP_DUPLICATE_LIB_OK%

echo.
echo ==========================================================
echo  SETUP COMPLETE. Next:
echo ==========================================================
echo  Backend A (opencode):  opencode auth login   (configure a provider)
echo  Backend B (llama-server, local GGUF):
echo     llama-server -m ^<path-to^>.gguf --port 6768 --host 127.0.0.1 -c 8192
echo     (a Qwen3-4B Q4_K_M GGUF works; keep LOCAL_LLM_BASE_URL pointed at :6768)
echo.
echo  Generate corpus:
echo     python run_phase15_corpus.py --dataset %DATASET% --graph %GRAPH% ^
echo        --out-dir %OUT_DIR% --mode harvest --corpus-file %CORPUS_FILE%
echo.
echo  Then build the V5 corpus + held-out metrics:
echo     python -m v5.training.corpus_scaling --corpus %OUT_DIR%\%CORPUS_FILE%
echo ==========================================================

if "%DO_RUN%"=="1" (
    echo.
    echo --run set: starting generation now ^(ensure a backend is serving first^)...
    python run_phase15_corpus.py --dataset "%DATASET%" --graph "%GRAPH%" --out-dir "%OUT_DIR%" --mode harvest --corpus-file "%CORPUS_FILE%"
)

endlocal
