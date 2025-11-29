#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de logging completo para o pipeline Kiss3DGen
Captura todas as mensagens, warnings e erros em arquivo
"""
import os
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_complete_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.DEBUG,
    capture_warnings: bool = True,
    capture_stdout: bool = True,
    log_name: Optional[str] = None
) -> tuple[logging.Logger, Path]:
    """
    Configura sistema de logging completo
    
    Args:
        log_dir: Diretório para salvar logs (default: outputs/logs)
        log_level: Nível de logging (default: DEBUG)
        capture_warnings: Capturar warnings do Python
        capture_stdout: Capturar stdout/stderr
        log_name: Nome do arquivo de log (default: timestamp)
    
    Returns:
        (logger, log_file_path)
    """
    if log_dir is None:
        log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if log_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_name = f"pipeline_{timestamp}.log"
    
    log_file = log_dir / log_name
    
    # Adicionar separador no início se o arquivo já existir (append mode)
    append_separator = log_file.exists()
    
    # Criar logger principal
    logger = logging.getLogger("kiss3dgen_pipeline")
    logger.setLevel(log_level)
    
    # Limpar handlers existentes
    logger.handlers.clear()
    
    # Handler para arquivo (completo) - usar append para não perder logs antigos
    file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Adicionar separador se estiver fazendo append
    if append_separator:
        logger.info("="*80)
        logger.info(f"CONTINUAÇÃO DE LOG - Nova execução iniciada em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
    
    # Handler para console (mais limpo)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Capturar warnings do Python
    if capture_warnings:
        logging.captureWarnings(True)
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.addHandler(file_handler)
        warnings_logger.addHandler(console_handler)
        warnings_logger.setLevel(logging.WARNING)
        
        # Redirecionar warnings.showwarning para logging
        def warning_to_log(message, category, filename, lineno, file=None, line=None):
            warnings_logger.warning(
                f"{category.__name__}: {message} (em {filename}:{lineno})"
            )
        warnings.showwarning = warning_to_log
    
    # Configurar loggers de bibliotecas importantes
    loggers_to_configure = [
        'transformers',
        'diffusers',
        'torch',
        'torchvision',
        'pytorch3d',
        'huggingface_hub',
        'PIL',
    ]
    
    for lib_name in loggers_to_configure:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.WARNING)  # Só warnings e erros por padrão
        lib_logger.addHandler(file_handler)
        lib_logger.propagate = False
    
    logger.info("="*80)
    logger.info("SISTEMA DE LOGGING INICIALIZADO")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Capture warnings: {capture_warnings}")
    logger.info("="*80)
    
    return logger, log_file

class LoggingContext:
    """Context manager para logging de seções específicas"""
    def __init__(self, logger: logging.Logger, section_name: str, level: int = logging.INFO):
        self.logger = logger
        self.section_name = section_name
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f">>> INICIANDO: {self.section_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self.logger.log(self.level, f"<<< CONCLUÍDO: {self.section_name} (tempo: {elapsed:.2f}s)")
        else:
            self.logger.error(
                f"<<< ERRO EM: {self.section_name} (tempo: {elapsed:.2f}s) - {exc_type.__name__}: {exc_val}"
            )
        return False  # Não suprime exceções

def log_model_operation(logger: logging.Logger, operation: str, model_name: str, device: str = None, memory_mb: float = None):
    """Log de operações com modelos"""
    msg = f"[MODEL] {operation}: {model_name}"
    if device:
        msg += f" | Device: {device}"
    if memory_mb is not None:
        msg += f" | Memory: {memory_mb:.1f} MB"
    logger.info(msg)

def log_memory_usage(logger: logging.Logger, label: str = "Memory check"):
    """Log de uso de memória GPU e CPU"""
    try:
        import torch
        import psutil
        import os
        
        # Memória GPU
        gpu_info = ""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            free = total_memory - reserved
            gpu_info = (
                f"GPU: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
                f"Max={max_allocated:.2f}GB, Free={free:.2f}GB, Total={total_memory:.2f}GB"
            )
        
        # Memória CPU/RAM
        process = psutil.Process(os.getpid())
        ram_info = process.memory_info()
        ram_used_gb = ram_info.rss / 1024**3  # GB
        ram_percent = process.memory_percent()
        system_memory = psutil.virtual_memory()
        ram_total_gb = system_memory.total / 1024**3  # GB
        ram_available_gb = system_memory.available / 1024**3  # GB
        
        cpu_info = (
            f"RAM: Process={ram_used_gb:.2f}GB ({ram_percent:.1f}%), "
            f"System={ram_available_gb:.2f}GB/{ram_total_gb:.2f}GB available"
        )
        
        logger.info(f"[MEMORY] {label} | {gpu_info} | {cpu_info}")
        
    except ImportError:
        # Fallback se psutil não estiver disponível
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
                logger.info(
                    f"[MEMORY] {label} | "
                    f"GPU: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Total={total_memory:.2f}GB"
                )
        except Exception as e:
            logger.warning(f"[MEMORY] Erro ao verificar memória: {e}")
    except Exception as e:
        logger.warning(f"[MEMORY] Erro ao verificar memória: {e}")

