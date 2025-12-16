# PowerShell script to run integration tests with real OpenAI API
#
# USAGE:
#   1. Set your OpenAI API key:
#      $env:OPENAI_API_KEY = "sk-..."
#   
#   2. Run this script:
#      .\run_integration_tests.ps1
#
#   3. Or run specific test:
#      .\run_integration_tests.ps1 -TestName "test_real_openai_function_calling"

param(
    [string]$TestName = "",
    [string]$Model = "gpt-4o-2024-08-06"
)

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "Integration Tests for Workflow Function Calling" -ForegroundColor Cyan
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host ""

# Check if API key is set
if (-not $env:OPENAI_API_KEY) {
    Write-Host "[ERROR] OPENAI_API_KEY environment variable not set!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please set it first:" -ForegroundColor Yellow
    Write-Host '  $env:OPENAI_API_KEY = "sk-your-key-here"' -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "[OK] OPENAI_API_KEY is set" -ForegroundColor Green
Write-Host "[INFO] Using model: $Model" -ForegroundColor Cyan
Write-Host ""

# Enable integration tests
$env:RUN_INTEGRATION_TESTS = "true"
$env:OPENAI_MODEL = $Model

# Build test path
$testPath = "tests/integration_tests/test_workflow_function_calling_integration.py"
if ($TestName) {
    $testPath += "::$TestName"
}

Write-Host "Running integration tests..." -ForegroundColor Cyan
Write-Host "Test path: $testPath" -ForegroundColor Gray
Write-Host ""

# Run tests
python -m pytest $testPath -v -s --tb=short

$exitCode = $LASTEXITCODE

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host "ALL INTEGRATION TESTS PASSED!" -ForegroundColor Green
} else {
    Write-Host "SOME INTEGRATION TESTS FAILED!" -ForegroundColor Red
}

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*79) -ForegroundColor Cyan

exit $exitCode

