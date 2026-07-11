#!/bin/bash

# Interrompe lo script immediatamente se un comando fallisce (ottimo per evitare di sprecare ore se c'è un typo)
set -e

# Imposta il numero di thread per tutti i run successivi
export THREADS=8

echo "========================================="
echo " Inizio Benchmark Suite - BAORECON"
echo " Thread impostati a: $THREADS"
echo "========================================="

echo ""
echo "[1/4] Esecuzione IFFT: CPU (SciPy) vs Reference vs GPU..."
python bench_bao_reconstructor.py --solver ifft --fft scipy --repeats 5

echo ""
echo "[2/4] Esecuzione IFFT: CPU (PyFFTW) vs GPU (Skip Reference)..."
python bench_bao_reconstructor.py --solver ifft --fft pyfftw --repeats 5 --skip_pyrecon

echo ""
echo "[3/4] Esecuzione Multigrid: Jacobi vs Reference..."
python bench_bao_reconstructor.py --solver multigrid --smoother jacobi --repeats 5

echo ""
echo "[4/4] Esecuzione Multigrid: MCGS (Skip Reference)..."
python bench_bao_reconstructor.py --solver multigrid --smoother mcgs --repeats 5 --skip_pyrecon

echo ""
echo "========================================="
echo " Benchmark completati con successo!"
echo "========================================="