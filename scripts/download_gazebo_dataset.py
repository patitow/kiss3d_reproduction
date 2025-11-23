#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para baixar o dataset do Google Research do Gazebo.
Baixa pelo menos 200 objetos com suas imagens e modelos 3D.

Desenvolvido para o projeto Mesh3D Generator - Visão Computacional 2025.2
Autor: Auto (Cursor AI Assistant)
Data: 2025
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import zipfile
import shutil
import re

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("[AVISO] beautifulsoup4 não instalado. Instale com: pip install beautifulsoup4")


class GazeboDatasetDownloader:
    """Downloader para o dataset do Google Research do Gazebo."""
    
    BASE_URL = "https://app.gazebosim.org"
    API_BASE = "https://fuel.gazebosim.org"
    GOOGLE_RESEARCH_OWNER = "GoogleResearch"
    
    def __init__(self, output_dir: str = "data/raw/gazebo_dataset", max_objects: int = 200):
        """
        Inicializa o downloader.
        
        Args:
            output_dir: Diretório de saída para os dados
            max_objects: Número máximo de objetos para baixar
        """
        self.output_dir = Path(output_dir)
        self.max_objects = max_objects
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Estrutura de diretórios
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
        
        self.downloaded_count = 0
        self.failed_count = 0
        
    def get_model_list(self, page: int = 1, per_page: int = 100) -> Optional[Dict]:
        """
        Obtém lista de modelos do GoogleResearch via API.
        
        Args:
            page: Número da página
            per_page: Itens por página
            
        Returns:
            Dicionário com lista de modelos ou None
        """
        try:
            # Tentar API do Fuel
            api_url = f"{self.API_BASE}/1.0/models"
            params = {
                'owner': self.GOOGLE_RESEARCH_OWNER,
                'page': page,
                'per_page': per_page
            }
            
            response = self.session.get(api_url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[AVISO] API retornou status {response.status_code}, tentando método alternativo...")
                return None
                
        except Exception as e:
            print(f"[AVISO] Erro ao acessar API: {e}")
            return None
    
    def get_model_list_from_web(self) -> List[str]:
        """
        Obtém lista de modelos do GoogleResearch usando scraping da página de busca.
        
        Returns:
            Lista de nomes de modelos
        """
        return self._get_model_list_scraping()
    
    def _get_model_list_scraping(self) -> List[str]:
        """Método de scraping da página web do Gazebo."""
        models = []
        page = 1
        
        print(f"[INFO] Buscando modelos do GoogleResearch na página do Gazebo...")
        
        while len(models) < self.max_objects:
            try:
                # URL de busca do Gazebo (formato correto: /search;q=termo)
                search_url = f"{self.BASE_URL}/search;q={self.GOOGLE_RESEARCH_OWNER}"
                
                response = self.session.get(search_url, timeout=30)
                
                if response.status_code != 200:
                    print(f"[AVISO] URL direta retornou {response.status_code}, tentando com parâmetros...")
                    # Tentar com parâmetros GET
                    search_url = f"{self.BASE_URL}/search"
                    params = {'q': self.GOOGLE_RESEARCH_OWNER}
                    response = self.session.get(search_url, params=params, timeout=30)
                    if response.status_code != 200:
                        print(f"[ERRO] Erro ao acessar página: {response.status_code}")
                        break
                
                content = response.text
                
                # Extrair nomes de modelos usando regex
                # Padrões possíveis:
                # /GoogleResearch/models/NOME_DO_MODELO
                # href="/GoogleResearch/models/NOME"
                patterns = [
                    r'/GoogleResearch/models/([^"\'<>/\s?&]+)',
                    r'href="[^"]*GoogleResearch/models/([^"\'<>/\s?&]+)',
                    r'url.*GoogleResearch/models/([^"\'<>/\s?&]+)',
                ]
                
                found_models = set()
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    found_models.update(matches)
                
                # Remover duplicatas e URLs codificadas
                import urllib.parse
                for model_name in found_models:
                    # Decodificar URL encoding
                    model_name = urllib.parse.unquote(model_name)
                    # Limpar caracteres especiais
                    model_name = model_name.strip()
                    if model_name and model_name not in models and len(model_name) > 1:
                        models.append(model_name)
                        if len(models) >= self.max_objects:
                            break
                
                if not found_models:
                    print(f"[INFO] Nenhum modelo encontrado. Tentando método alternativo...")
                    # Tentar acessar diretamente a página do owner
                    owner_url = f"{self.BASE_URL}/{self.GOOGLE_RESEARCH_OWNER}/models"
                    owner_response = self.session.get(owner_url, timeout=30)
                    if owner_response.status_code == 200:
                        content = owner_response.text
                        for pattern in patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            found_models.update(matches)
                        for model_name in found_models:
                            model_name = urllib.parse.unquote(model_name).strip()
                            if model_name and model_name not in models and len(model_name) > 1:
                                models.append(model_name)
                                if len(models) >= self.max_objects:
                                    break
                    break
                
                print(f"[INFO] Encontrados {len(models)} modelos únicos até agora...")
                
                # Se encontrou modelos mas não atingiu o limite, tentar próxima página
                if len(models) < self.max_objects and found_models:
                    # Verificar se há botão de próxima página
                    if 'Next page' in content and 'disabled' not in content.lower():
                        page += 1
                        time.sleep(1)
                    else:
                        break
                else:
                    break
                    
            except Exception as e:
                print(f"[ERRO] Erro no scraping: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Remover modelos inválidos (que não são do GoogleResearch)
        valid_models = [m for m in models if m and not m.startswith('http') and len(m) > 1]
        
        print(f"[OK] Total de {len(valid_models)} modelos encontrados")
        return valid_models[:self.max_objects]
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Obtém informações de um modelo específico.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Dicionário com informações do modelo ou None
        """
        try:
            # Tentar API
            api_url = f"{self.API_BASE}/1.0/{self.GOOGLE_RESEARCH_OWNER}/models/{model_name}"
            response = self.session.get(api_url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                # Fallback: construir URL da página
                page_url = f"{self.BASE_URL}/{self.GOOGLE_RESEARCH_OWNER}/models/{model_name}"
                return {'url': page_url, 'name': model_name}
                
        except Exception as e:
            print(f"[AVISO] Erro ao obter info do modelo {model_name}: {e}")
            return None
    
    def download_model(self, model_name: str) -> bool:
        """
        Baixa um modelo completo (arquivos 3D, imagens, metadados).
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            True se download foi bem-sucedido
        """
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        print(f"[INFO] Baixando modelo: {model_name}")
        
        try:
            # 1. Obter informações do modelo
            model_info = self.get_model_info(model_name)
            if not model_info:
                print(f"[ERRO] Não foi possível obter informações do modelo {model_name}")
                return False
            
            # Salvar metadados
            metadata_file = self.metadata_dir / f"{model_name}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            # 2. Tentar baixar via API do Fuel
            download_url = None
            if 'download_url' in model_info:
                download_url = model_info['download_url']
            elif 'files' in model_info and model_info['files']:
                # Tentar encontrar URL de download
                for file_info in model_info['files']:
                    if file_info.get('file_type') == 'model' or 'zip' in file_info.get('file_path', ''):
                        download_url = file_info.get('download_url')
                        break
            
            # 3. Se não tiver URL direta, tentar construir URLs possíveis
            if not download_url:
                # Tentar diferentes formatos de URL do Fuel
                possible_urls = [
                    f"{self.API_BASE}/1.0/{self.GOOGLE_RESEARCH_OWNER}/models/{model_name}/tip/files",
                    f"{self.API_BASE}/1.0/{self.GOOGLE_RESEARCH_OWNER}/models/{model_name}/files",
                    f"{self.BASE_URL}/{self.GOOGLE_RESEARCH_OWNER}/models/{model_name}/files",
                ]
                
                for url in possible_urls:
                    try:
                        test_response = self.session.head(url, timeout=10, allow_redirects=True)
                        if test_response.status_code == 200:
                            download_url = url
                            break
                    except:
                        continue
            
            # 4. Baixar arquivos do modelo
            if download_url:
                try:
                    response = self.session.get(download_url, stream=True, timeout=60)
                    
                    if response.status_code == 200:
                        # Salvar como ZIP
                        zip_path = model_dir / f"{model_name}.zip"
                        with open(zip_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        
                        # Extrair ZIP
                        try:
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(model_dir)
                            zip_path.unlink()  # Remover ZIP após extrair
                            print(f"[OK] Modelo {model_name} baixado e extraído")
                        except zipfile.BadZipFile:
                            print(f"[AVISO] Arquivo não é ZIP válido, mantendo como está")
                        
                        # Procurar por imagens no diretório extraído
                        self._extract_images(model_dir, model_name)
                        
                        return True
                    else:
                        print(f"[AVISO] Download retornou status {response.status_code}")
                except Exception as e:
                    print(f"[AVISO] Erro ao baixar arquivos: {e}")
            
            # 5. Fallback: tentar baixar página e extrair links
            page_url = f"{self.BASE_URL}/{self.GOOGLE_RESEARCH_OWNER}/models/{model_name}"
            response = self.session.get(page_url, timeout=30)
            
            if response.status_code == 200:
                # Salvar página HTML como fallback
                html_path = model_dir / "page.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Tentar extrair links de imagens
                import re
                img_patterns = [
                    r'src="([^"]+\.(jpg|jpeg|png|gif))"',
                    r'href="([^"]+\.(jpg|jpeg|png|gif))"',
                ]
                
                for pattern in img_patterns:
                    matches = re.findall(pattern, response.text, re.IGNORECASE)
                    for match in matches[:5]:  # Limitar a 5 imagens
                        img_url = match[0] if isinstance(match, tuple) else match
                        if not img_url.startswith('http'):
                            img_url = urljoin(page_url, img_url)
                        self._download_image(img_url, model_name)
                
                print(f"[OK] Modelo {model_name} processado (modo fallback)")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERRO] Erro ao baixar modelo {model_name}: {e}")
            return False
    
    def _extract_images(self, model_dir: Path, model_name: str):
        """Extrai imagens do diretório do modelo."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        for ext in image_extensions:
            for img_file in model_dir.rglob(f'*{ext}'):
                if img_file.is_file():
                    # Copiar para diretório de imagens
                    dest = self.images_dir / f"{model_name}_{img_file.name}"
                    shutil.copy2(img_file, dest)
    
    def _download_image(self, img_url: str, model_name: str):
        """Baixa uma imagem individual."""
        try:
            response = self.session.get(img_url, timeout=30)
            if response.status_code == 200:
                # Determinar extensão
                ext = Path(urlparse(img_url).path).suffix or '.jpg'
                img_path = self.images_dir / f"{model_name}{ext}"
                
                with open(img_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            pass  # Ignorar erros de imagens individuais
    
    def download_dataset(self):
        """Baixa o dataset completo."""
        print("=" * 60)
        print("[INFO] Iniciando download do dataset GoogleResearch")
        print(f"[INFO] Objetivo: {self.max_objects} objetos")
        print("=" * 60)
        
        # Tentar usar a versão melhorada primeiro
        try:
            from scripts.download_gazebo_dataset_v2 import GazeboDatasetDownloaderV2
            print("[INFO] Usando versão melhorada do downloader...")
            downloader_v2 = GazeboDatasetDownloaderV2(
                output_dir=str(self.output_dir),
                max_objects=self.max_objects
            )
            downloader_v2.download_dataset()
            return
        except Exception as e:
            print(f"[AVISO] Versão melhorada não disponível: {e}")
            print("[INFO] Usando método padrão...")
        
        # Obter lista de modelos
        models = self.get_model_list_from_web()
        
        if not models:
            print("[ERRO] Não foi possível obter lista de modelos")
            return
        
        print(f"[OK] Encontrados {len(models)} modelos para baixar")
        
        # Baixar cada modelo
        for i, model_name in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Processando: {model_name}")
            
            if self.download_model(model_name):
                self.downloaded_count += 1
            else:
                self.failed_count += 1
            
            # Rate limiting
            time.sleep(0.5)
            
            # Salvar progresso
            if i % 10 == 0:
                self._save_progress(models, i)
        
        # Resumo final
        print("\n" + "=" * 60)
        print("[RESUMO] Download concluído!")
        print("=" * 60)
        print(f"Baixados com sucesso: {self.downloaded_count}")
        print(f"Falhas: {self.failed_count}")
        print(f"Total processado: {len(models)}")
        print(f"\nDataset salvo em: {self.output_dir.absolute()}")
    
    def _save_progress(self, models: List[str], current_index: int):
        """Salva progresso do download."""
        progress_file = self.output_dir / "download_progress.json"
        progress = {
            'total_models': len(models),
            'processed': current_index,
            'downloaded': self.downloaded_count,
            'failed': self.failed_count,
            'models': models[:current_index]
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Baixa dataset do Google Research do Gazebo"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/gazebo_dataset',
        help='Diretório de saída (padrão: data/raw/gazebo_dataset)'
    )
    parser.add_argument(
        '--max-objects',
        type=int,
        default=200,
        help='Número máximo de objetos para baixar (padrão: 200)'
    )
    
    args = parser.parse_args()
    
    downloader = GazeboDatasetDownloader(
        output_dir=args.output,
        max_objects=args.max_objects
    )
    
    downloader.download_dataset()


if __name__ == '__main__':
    main()

