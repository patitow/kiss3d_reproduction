#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coleta nomes de modelos do GoogleResearch via scraping das paginas."""

import json
import time
import requests
import re
from pathlib import Path
from typing import Set

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class ModelNameCollector:
    """Coleta nomes de modelos navegando pelas paginas."""
    
    BASE_URL = "https://app.gazebosim.org"
    OWNER = "GoogleResearch"
    MODELS_PER_PAGE = 20
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def collect_all_names(self, max_models: int = 200) -> list:
        """Coleta nomes navegando pelas paginas."""
        models = set()
        page = 1
        
        print(f"[INFO] Coletando nomes de modelos do {self.OWNER}...")
        print(f"[INFO] Objetivo: {max_models} modelos")
        print(f"[INFO] Navegando pelas paginas (20 por pagina)...\n")
        
        while len(models) < max_models:
            try:
                # URL da página (formato correto do Gazebo)
                url = f"{self.BASE_URL}/{self.OWNER}/fuel/models"
                if page > 1:
                    url += f"?page={page}"
                
                print(f"[INFO] Acessando pagina {page}...", end=" ", flush=True)
                response = self.session.get(url, timeout=30)
                
                if response.status_code != 200:
                    print(f"Erro {response.status_code}")
                    break
                
                # Extrair nomes de modelos
                page_models = self._extract_model_names(response.text)
                
                if page_models:
                    models.update(page_models)
                    print(f"✓ {len(page_models)} modelos encontrados | Total: {len(models)}")
                else:
                    print("Nenhum modelo encontrado")
                
                # Verificar se há próxima página
                if not self._has_next_page(response.text, page):
                    print(f"\n[INFO] Ultima pagina alcancada")
                    break
                
                if len(models) >= max_models:
                    break
                
                page += 1
                time.sleep(0.5)
                    
            except Exception as e:
                print(f"\n[ERRO] Erro na pagina {page}: {e}")
                break
        
        models_list = sorted(list(models))[:max_models]
        print(f"\n[OK] Total de {len(models_list)} nomes coletados")
        return models_list
    
    def _extract_model_names(self, html: str) -> Set[str]:
        """Extrai nomes de modelos do HTML."""
        models = set()
        
        if HAS_BS4:
            soup = BeautifulSoup(html, 'html.parser')
            links = soup.find_all('a', href=re.compile(rf'/{self.OWNER}/models/[^/"]+'))
            for link in links:
                href = link.get('href', '')
                match = re.search(rf'/{self.OWNER}/models/([^/"]+)', href)
                if match:
                    name = match.group(1)
                    if name and name != self.OWNER and len(name) > 1:
                        models.add(name)
        else:
            pattern = rf'/{self.OWNER}/models/([^/"\'<>\s?&]+)'
            matches = re.findall(pattern, html)
            models = {m for m in matches if m and m != self.OWNER and len(m) > 1}
        
        return models
    
    def _has_next_page(self, html: str, current_page: int) -> bool:
        """Verifica se ha proxima pagina."""
        # Verificar padrão de paginação
        page_info = re.search(r'(\d+)\s*[–-]\s*(\d+)\s+of\s+(\d+)', html)
        if page_info:
            start, end, total = map(int, page_info.groups())
            return end < total
        
        # Verificar botão Next page
        if 'Next page' in html:
            # Se não está disabled, há próxima página
            next_button_disabled = 'Next page' in html and 'disabled' in html.lower()
            return not next_button_disabled
        
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Coleta nomes de modelos do Gazebo")
    parser.add_argument('--max-models', type=int, default=200, help='Numero maximo de modelos')
    parser.add_argument('--output', default='model_names.json', help='Arquivo de saida JSON')
    
    args = parser.parse_args()
    
    collector = ModelNameCollector()
    model_names = collector.collect_all_names(max_models=args.max_models)
    
    # Salvar lista
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(model_names),
            'models': model_names
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Lista salva em: {output_path.absolute()}")
    print(f"[INFO] Primeiros 10 modelos:")
    for i, name in enumerate(model_names[:10], 1):
        print(f"  {i}. {name}")


if __name__ == '__main__':
    main()

