#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para remover acentos de todos os arquivos Python do projeto
"""

import os
import re
from pathlib import Path

# Mapeamento de caracteres acentuados para não acentuados
ACCENT_MAP = {
    'a': 'a', 'a': 'a', 'a': 'a', 'a': 'a', 'a': 'a',
    'e': 'e', 'e': 'e', 'e': 'e', 'e': 'e',
    'i': 'i', 'i': 'i', 'i': 'i', 'i': 'i',
    'o': 'o', 'o': 'o', 'o': 'o', 'o': 'o', 'o': 'o',
    'u': 'u', 'u': 'u', 'u': 'u', 'u': 'u',
    'c': 'c',
    'A': 'A', 'A': 'A', 'A': 'A', 'A': 'A', 'A': 'A',
    'E': 'E', 'E': 'E', 'E': 'E', 'E': 'E',
    'I': 'I', 'I': 'I', 'I': 'I', 'I': 'I',
    'O': 'O', 'O': 'O', 'O': 'O', 'O': 'O', 'O': 'O',
    'U': 'U', 'U': 'U', 'U': 'U', 'U': 'U',
    'C': 'C',
}

def remove_accents(text):
    """Remove acentos de um texto"""
    for accented, unaccented in ACCENT_MAP.items():
        text = text.replace(accented, unaccented)
    
    # Remover emojis problemáticos e substituir por texto simples
    emoji_replacements = {
        '[AVISO]': '[AVISO]',
        '[AVISO]': '[AVISO]',
        '[OK]': '[OK]',
        '[ERRO]': '[ERRO]',
        '🔧': '',
        '📊': '',
        '🎯': '',
        '📝': '',
        '🚀': '',
    }
    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)
    
    return text

def process_file(file_path):
    """Processa um arquivo Python removendo acentos"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remover acentos apenas em strings (dentro de aspas)
        # Preservar comentários e docstrings
        lines = content.split('\n')
        new_lines = []
        changed = False
        
        for line in lines:
            # Verificar se a linha contém strings com acentos
            # Processar strings entre aspas simples ou duplas
            new_line = line
            if any(accent in line for accent in ACCENT_MAP.keys()):
                # Remover acentos em strings
                # Usar regex para encontrar strings
                def replace_in_string(match):
                    string_content = match.group(1)
                    return match.group(0)[0] + remove_accents(string_content) + match.group(0)[-1]
                
                # Processar strings simples
                new_line = re.sub(r'(["\'])(.*?)(\1)', lambda m: m.group(1) + remove_accents(m.group(2)) + m.group(3), new_line)
                
                if new_line != line:
                    changed = True
            
            new_lines.append(new_line)
        
        if changed:
            new_content = '\n'.join(new_lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return False

def main():
    """Processa todos os arquivos Python do projeto"""
    project_root = Path(__file__).parent.parent
    
    # Diretórios a processar
    dirs_to_process = [
        project_root / 'mesh3d_generator',
        project_root / 'scripts',
    ]
    
    files_processed = 0
    files_changed = 0
    
    for dir_path in dirs_to_process:
        if not dir_path.exists():
            continue
        
        for py_file in dir_path.rglob('*.py'):
            files_processed += 1
            if process_file(py_file):
                files_changed += 1
                print(f"Processado: {py_file.relative_to(project_root)}")
    
    print(f"\nTotal de arquivos processados: {files_processed}")
    print(f"Arquivos modificados: {files_changed}")

if __name__ == '__main__':
    main()

