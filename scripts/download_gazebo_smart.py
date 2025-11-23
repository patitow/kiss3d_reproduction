#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download inteligente de modelos 3D do Google Research do Gazebo.

Estratégia:
1. Scraping da página do owner para obter lista completa de nomes
2. Usa API do Fuel para baixar cada modelo individualmente
3. Garante 200 modelos baixados
"""

import json
import time
import requests
from pathlib import Path
from typing import List, Set
from urllib.parse import quote
import re

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("[AVISO] beautifulsoup4 não instalado. Instale com: pip install beautifulsoup4")


class SmartGazeboDownloader:
    """Downloader inteligente com scraping + API."""
    
    BASE_URL = "https://app.gazebosim.org"
    FUEL_API = "https://fuel.gazebosim.org"
    OWNER = "GoogleResearch"
    MODELS_PER_PAGE = 20
    
    def __init__(self, output_dir: str = "data/raw/gazebo_dataset", max_models: int = 200):
        self.output_dir = Path(output_dir)
        self.max_models = max_models
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.models_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.downloaded = 0
        self.failed = 0
    
    def scrape_all_model_names(self) -> List[str]:
        """Faz scraping navegando pelas páginas para obter lista completa de nomes."""
        models = set()
        page = 1
        
        print(f"[INFO] Fazendo scraping de modelos do {self.OWNER}...")
        print(f"[INFO] Objetivo: {self.max_models} modelos")
        print(f"[INFO] Navegando pelas páginas (20 modelos por página)...\n")
        
        while len(models) < self.max_models:
            try:
                # URL da página (página 1 não precisa de parâmetro)
                if page == 1:
                    url = f"{self.BASE_URL}/{self.OWNER}/fuel/models"
                else:
                    url = f"{self.BASE_URL}/{self.OWNER}/fuel/models?page={page}"
                
                response = self.session.get(url, timeout=30)
                
                if response.status_code != 200:
                    print(f"[AVISO] Erro {response.status_code} na página {page}")
                    break
                
                # Extrair nomes de modelos da página atual
                page_models = set()
                
                if HAS_BS4:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Procurar todos os links de modelos
                    links = soup.find_all('a', href=re.compile(rf'/{self.OWNER}/models/[^/"]+'))
                    for link in links:
                        href = link.get('href', '')
                        # Extrair nome do modelo (não o owner)
                        match = re.search(rf'/{self.OWNER}/models/([^/"]+)', href)
                        if match:
                            model_name = match.group(1)
                            # Filtrar: não pode ser o próprio owner nem vazio
                            if model_name and model_name != self.OWNER and len(model_name) > 1:
                                page_models.add(model_name)
                else:
                    # Fallback: regex mais robusto
                    pattern = rf'/{self.OWNER}/models/([^/"\'<>\s?&]+)'
                    matches = re.findall(pattern, response.text)
                    page_models = {m for m in matches if m and m != self.OWNER and len(m) > 1}
                
                if page_models:
                    models.update(page_models)
                    print(f"[INFO] Página {page}: {len(page_models)} modelos encontrados | Total acumulado: {len(models)}")
                else:
                    print(f"[AVISO] Página {page}: Nenhum modelo encontrado")
                
                # Verificar se há próxima página
                # Procurar por "Next page" button que não está disabled
                has_next = False
                if 'Next page' in response.text:
                    # Verificar se o botão não está disabled
                    if 'disabled' not in response.text.lower() or 'Next page' in response.text:
                        # Verificar se há mais itens
                        page_info_match = re.search(r'(\d+)\s*–\s*(\d+)\s+of\s+(\d+)', response.text)
                        if page_info_match:
                            start, end, total = map(int, page_info_match.groups())
                            if end < total and len(models) < self.max_models:
                                has_next = True
                        else:
                            # Tentar padrão alternativo
                            if len(page_models) == self.MODELS_PER_PAGE:
                                has_next = True
                
                if not has_next or len(models) >= self.max_models:
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting
                    
            except Exception as e:
                print(f"[ERRO] Erro na página {page}: {e}")
                break
        
        models_list = sorted(list(models))[:self.max_models]
        print(f"\n[OK] {len(models_list)} modelos únicos obtidos via scraping")
        return models_list
    
    def download_model(self, model_name: str) -> bool:
        """Baixa um modelo usando a API do Fuel."""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        try:
            model_encoded = quote(model_name, safe='')
            
            # 1. Obter metadados
            info_url = f"{self.FUEL_API}/1.0/{self.OWNER}/models/{model_encoded}"
            try:
                info_resp = self.session.get(info_url, timeout=30)
                if info_resp.status_code == 200:
                    metadata = info_resp.json()
                    with open(self.metadata_dir / f"{model_name}.json", 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)
            except:
                pass
            
            # 2. Obter file_tree
            files_url = f"{self.FUEL_API}/1.0/{self.OWNER}/models/{model_encoded}/tip/files"
            files_resp = self.session.get(files_url, timeout=30)
            
            if files_resp.status_code != 200:
                return False
            
            files_data = files_resp.json()
            file_tree = files_data.get('file_tree', [])
            
            if not file_tree:
                return False
            
            # 3. Baixar todos os arquivos
            downloaded_files = 0
            
            def process_node(node):
                nonlocal downloaded_files
                if 'children' in node:
                    for child in node['children']:
                        process_node(child)
                else:
                    file_path = node.get('path', '')
                    if file_path:
                        file_url = f"{self.FUEL_API}/1.0/{self.OWNER}/models/{model_encoded}/tip/files{file_path}"
                        try:
                            file_resp = self.session.get(file_url, timeout=30, stream=True)
                            if file_resp.status_code == 200:
                                local_path = model_dir / file_path.lstrip('/')
                                local_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                with open(local_path, 'wb') as f:
                                    for chunk in file_resp.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                
                                downloaded_files += 1
                                
                                # Extrair imagens
                                if file_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                                    import shutil
                                    dest = self.images_dir / f"{model_name}_{Path(file_path).name}"
                                    shutil.copy2(local_path, dest)
                        except:
                            pass
            
            for node in file_tree:
                process_node(node)
            
            return downloaded_files > 0
            
        except Exception as e:
            return False
    
    def download_all(self):
        """Executa o download completo."""
        print("=" * 60)
        print(f"[INFO] Download Inteligente - {self.OWNER}")
        print(f"[INFO] Objetivo: {self.max_models} modelos")
        print("=" * 60)
        
        # 1. Obter lista de nomes via scraping
        models = self.scrape_all_model_names()
        
        if not models:
            print("[ERRO] Nenhum modelo encontrado!")
            return
        
        print(f"\n[OK] {len(models)} modelos para baixar\n")
        
        # 2. Baixar cada modelo
        for i, model_name in enumerate(models, 1):
            print(f"[{i}/{len(models)}] ", end="", flush=True)
            
            if self.download_model(model_name):
                self.downloaded += 1
                print(f"[OK] {model_name}")
            else:
                self.failed += 1
                print(f"[FALHOU] {model_name}")
            
            time.sleep(0.3)  # Rate limiting
            
            # Salvar progresso
            if i % 10 == 0:
                self._save_progress(models, i)
        
        # Resumo
        print("\n" + "=" * 60)
        print("[RESUMO] Download concluído!")
        print("=" * 60)
        print(f"Baixados: {self.downloaded}/{len(models)}")
        print(f"Falhas: {self.failed}")
        print(f"\nDataset em: {self.output_dir.absolute()}")
    
    def _save_progress(self, models: List[str], current: int):
        """Salva progresso."""
        progress = {
            'total': len(models),
            'processed': current,
            'downloaded': self.downloaded,
            'failed': self.failed
        }
        with open(self.output_dir / "download_progress.json", 'w') as f:
            json.dump(progress, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download inteligente de modelos Gazebo")
    parser.add_argument('--max-objects', type=int, default=200, help='Número de modelos')
    parser.add_argument('--output', default='data/raw/gazebo_dataset', help='Diretório de saída')
    
    args = parser.parse_args()
    
    downloader = SmartGazeboDownloader(
        output_dir=args.output,
        max_models=args.max_objects
    )
    
    downloader.download_all()


if __name__ == '__main__':
    main()

