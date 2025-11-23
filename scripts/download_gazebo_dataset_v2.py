#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download de modelos 3D do Google Research do Gazebo."""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse, quote
import zipfile
import shutil
import re


class GazeboDatasetDownloaderV2:
    """Downloader melhorado para o dataset do Google Research do Gazebo."""
    
    BASE_URL = "https://app.gazebosim.org"
    FUEL_API = "https://fuel.gazebosim.org"
    GOOGLE_RESEARCH_OWNER = "GoogleResearch"
    
    def __init__(self, output_dir: str = "data/raw/gazebo_dataset", max_objects: int = 200):
        """Inicializa o downloader."""
        self.output_dir = Path(output_dir)
        self.max_objects = max_objects
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = self.output_dir / "models"
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.models_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        self.downloaded_count = 0
        self.failed_count = 0
        
        # Lista completa de modelos (fornecida pelo usuário)
        self.known_models = self._get_known_models_list()
    
    def _get_known_models_list(self) -> List[str]:
        """Retorna lista completa de modelos do GoogleResearch."""
        return [
            "Weisshai_Great_White_Shark", "Vtech_Stack_Sing_Rings_636_Months", "Vtech_Roll_Learn_Turtle",
            "Vtech_Cruise_Learn_Car_25_Years", "Victor_Reversible_Bookend", "VEGETABLE_GARDEN",
            "Utana_5_Porcelain_Ramekin_Large", "Ubisoft_RockSmith_Real_Tone_Cable_Xbox_360",
            "TriStar_Products_PPC_Power_Pressure_Cooker_XL_in_Black", "Toysmith_Windem_Up_Flippin_Animals_Dog",
            "Toys_R_Us_Treat_Dispenser_Smart_Puzzle_Foobler", "Top_Paw_Dog_Bowl_Blue_Paw_Bone_Ceramic_25_fl_oz_total",
            "Top_Paw_Dog_Bow_Bone_Ceramic_13_fl_oz_total", "Threshold_Tray_Rectangle_Porcelain",
            "Threshold_Textured_Damask_Bath_Towel_Pink", "Threshold_Salad_Plate_Square_Rim_Porcelain",
            "Threshold_Ramekin_White_Porcelain", "Threshold_Porcelain_Teapot_White",
            "Threshold_Porcelain_Spoon_Rest_White", "Threshold_Porcelain_Serving_Bowl_Coupe_White",
            "Threshold_Porcelain_Pitcher_White", "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White",
            "Threshold_Performance_Bath_Sheet_Sandoval_Blue_33_x_63", "Threshold_Hand_Towel_Blue_Medallion_16_x_27",
            "Threshold_Dinner_Plate_Square_Rim_White_Porcelain", "Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring",
            "Threshold_Bead_Cereal_Bowl_White", "Threshold_Basket_Natural_Finish_Fabric_Liner_Small",
            "Threshold_Bamboo_Ceramic_Soap_Dish", "Target_Basket_Medium",
            "Tag_Dishtowel_Waffle_Gray_Checks_18_x_26", "Tag_Dishtowel_Green",
            "Tag_Dishtowel_Dobby_Stripe_Blue_18_x_26", "Tag_Dishtowel_Basket_Weave_Red_18_x_26",
            "Tag_Dishtowel_18_x_26", "TWIST_SHAPE", "TWISTED_PUZZLE_twb4AyFtu8Q", "TWISTED_PUZZLE",
            "TEA_SET", "TABLEWARE_SET_5ww1UFLuCJG", "TABLEWARE_SET_5CHkPjjxVpp", "TABLEWARE_SET",
            "Sushi_Mat", "Sterilite_Caddy_Blue_Sky_17_58_x_12_58_x_9_14", "Squirrel",
            "Spritz_Easter_Basket_Plastic_Teal", "Sootheze_Toasty_Orca", "Sootheze_Cold_Therapy_Elephant",
            "Sonny_School_Bus", "Smith_Hawken_Woven_BasketTray_Organizer_with_3_Compartments_95_x_9_x_13",
            "Shurtape_Tape_Purple_CP28", "Shurtape_Gaffers_Tape_Silver_2_x_60_yd",
            "Shurtape_30_Day_Removal_UV_Delct_15", "Shaxon_100_Molded_Category_6_RJ45RJ45_Shielded_Patch_Cord_White",
            "Shark", "Sea_to_Summit_Xl_Bowl", "Schleich_Therizinosaurus_ln9cruulPqc",
            "Schleich_Spinosaurus_Action_Figure", "Schleich_S_Bayala_Unicorn_70432",
            "Schleich_Lion_Action_Figure", "Schleich_Hereford_Bull", "Schleich_Bald_Eagle",
            "Schleich_Allosaurus", "Schleich_African_Black_Rhino",
            "Sapota_Threshold_4_Ceramic_Round_Planter_Red", "STEAK_SET", "STACKING_RING",
            "STACKING_BEAR_V04KKgGBn2A", "STACKING_BEAR", "SORTING_TRAIN", "SORTING_BUS",
            "SHAPE_SORTER", "SHAPE_MATCHING_NxacpAY9jDt", "SHAPE_MATCHING", "SAPPHIRE_R7_260X_OC",
            "SANDWICH_MEAL", "Rubbermaid_Large_Drainer", "Room_Essentials_Salad_Plate_Turquoise",
            "Room_Essentials_Mug_White_Yellow", "Room_Essentials_Kitchen_Towels_16_x_26_2_count",
            "Room_Essentials_Fabric_Cube_Lavender", "Room_Essentials_Dish_Drainer_Collapsible_White",
            "Room_Essentials_Bowl_Turquiose", "Rexy_Glove_Heavy_Duty_Large",
            "Rexy_Glove_Heavy_Duty_Gloves_Medium", "Retail_Leadership_Summit_tQFCizMt6g0",
            "Retail_Leadership_Summit_eCT3zqHYIkX", "Retail_Leadership_Summit",
            "Remington_TStudio_Hair_Dryer", "Reef_Star_Cushion_Flipflops_Size_8_Black",
            "Razer_Taipan_White_Ambidextrous_Gaming_Mouse", "Razer_Taipan_Black_Ambidextrous_Gaming_Mouse",
            "Razer_Naga_MMO_Gaming_Mouse", "Razer_Kraken_Pro_headset_Full_size_Black",
            "Razer_Kraken_71_Chroma_headset_Full_size_Black",
            "Razer_Goliathus_Control_Edition_Small_Soft_Gaming_Mouse_Mat",
            "Razer_Blackwidow_Tournament_Edition_Keyboard",
            "Razer_BlackWidow_Ultimate_2014_Mechanical_Gaming_Keyboard",
            "Razer_BlackWidow_Stealth_2014_Keyboard_07VFzIVabgh", "Razer_Abyssus_Ambidextrous_Gaming_Mouse",
            "Racoon", "RJ_Rabbit_Easter_Basket_Blue", "REEF_ZENFUN", "REEF_BRAIDED_CUSHION", "REEF_BANTU",
            "Provence_Bath_Towel_Royal_Blue", "Progressive_Rubber_Spatulas_3_count",
            "ProSport_Harness_to_Booster_Seat", "Poppin_File_Sorter_White", "Poppin_File_Sorter_Pink",
            "Poppin_File_Sorter_Blue", "Pony_C_Clamp_1440", "Perricone_MD_Vitamin_C_Ester_Serum",
            "Perricone_MD_Vitamin_C_Ester_15", "Perricone_MD_The_Power_Treatments",
            "Perricone_MD_The_Metabolic_Formula_Supplements", "Perricone_MD_The_Crease_Cure_Duo",
            "Perricone_MD_The_Cold_Plasma_Face_Eyes_Duo",
            "Perricone_MD_Super_Berry_Powder_with_Acai_Supplements",
            "Perricone_MD_Skin_Total_Body_Supplements", "Perricone_MD_Skin_Clear_Supplements",
            "Perricone_MD_Photo_Plasma", "Perricone_MD_Omega_3_Supplements", "Perricone_MD_OVM",
            "Perricone_MD_Nutritive_Cleanser", "Perricone_MD_No_Mascara_Mascara",
            "Perricone_MD_No_Lipstick_Lipstick", "Perricone_MD_No_Foundation_Serum",
            "Perricone_MD_No_Foundation_Foundation_No_1", "Perricone_MD_No_Bronzer_Bronzer",
            "Perricone_MD_Neuropeptide_Firming_Moisturizer",
            "Perricone_MD_Neuropeptide_Facial_Conformer",
            "Perricone_MD_Hypoallergenic_Gentle_Cleanser",
            "Perricone_MD_Hypoallergenic_Firming_Eye_Cream_05_oz",
            "Perricone_MD_High_Potency_Evening_Repair",
            "Perricone_MD_Health_Weight_Management_Supplements",
            "Perricone_MD_Firming_Neck_Therapy_Treatment",
            "Perricone_MD_Face_Finishing_Moisturizer_4_oz", "Perricone_MD_Face_Finishing_Moisturizer",
            "Perricone_MD_Cold_Plasma_Body", "Perricone_MD_Cold_Plasma", "Perricone_MD_Chia_Serum",
            "Perricone_MD_Blue_Plasma_Orbital",
            "Perricone_MD_Best_of_Perricone_7Piece_Collection_MEGsO6GIsyL",
            "Perricone_MD_AcylGlutathione_Eye_Lid_Serum",
            "Perricone_MD_AcylGlutathione_Deep_Crease_Serum", "Perricoen_MD_No_Concealer_Concealer",
            "Pepsi_Max_Cola_Zero_Calorie_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt",
            "Pepsi_Cola_Wild_Cherry_Diet_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt",
            "Pepsi_Cola_Caffeine_Free_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt",
            "Pepsi_Caffeine_Free_Diet_12_CT", "Pennington_Electric_Pot_Cabana_4",
            "PUNCH_DROP_TjicLPMqLvz", "PUNCH_DROP", "POUNDING_MUSHROOMS", "PEPSI_NEXT_CACRV",
            "PARENT_ROOM_FURNITURE_SET_1_DLKEy8H4mwK", "PARENT_ROOM_FURNITURE_SET_1",
            "Ortho_Forward_Facing_QCaor9ImJ2G", "Ortho_Forward_Facing_CkAW6rL25xH",
            "Ortho_Forward_Facing_3Q6J2oKJD92", "Ortho_Forward_Facing",
            "Ocedar_Snap_On_Dust_Pan_And_Brush_1_ct", "Object_REmvBDJStub", "Object",
            "OXO_Soft_Works_Can_Opener_SnapLock", "OXO_Cookie_Spatula",
            "Now_Designs_Snack_Bags_Bicycle_2_count", "Now_Designs_Dish_Towel_Mojave_18_x_28",
            "Now_Designs_Bowl_Akita_Black", "Nordic_Ware_Original_Bundt_Pan",
            "Neat_Solutions_Character_Bib_2_pack", "Markings_Letter_Holder", "Markings_Desk_Caddy",
            "Magnifying_Glassassrt", "MY_MOOD_MEMO",
            "Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige", "Kotobuki_Saucer_Dragon_Fly",
            "Kong_Puppy_Teething_Rubber_Small_Pink",
            "KS_Chocolate_Cube_Box_Assortment_By_Neuhaus_2010_Ounces",
            "KITCHEN_SET_CLASSIC_40HwCHfeG0H", "KITCHEN_FURNITURE_SET_1",
            "JA_Henckels_International_Premio_Cutlery_Block_Set_14Piece", "InterDesign_Over_Door",
            "In_Green_Company_Surface_Saver_Ring_10_Terra_Cotta",
            "INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count",
            "Home_Fashions_Washcloth_Olive_Green", "Home_Fashions_Washcloth_Linen", "Hilary",
            "Hefty_Waste_Basket_Decorative_Bronze_85_liter", "HeavyDuty_Flashlight",
            "Grreatv_Choice_Dog_Bowl_Gray_Bones_Plastic_20_fl_oz_total",
            "Grreat_Choice_Dog_Double_Dish_Plastic_Blue", "Great_Dinos_Triceratops_Toy",
            "Granimals_20_Wooden_ABC_Blocks_Wagon_g2TinmUGGHI",
            "Granimals_20_Wooden_ABC_Blocks_Wagon_85VdSftGsLi",
            "Granimals_20_Wooden_ABC_Blocks_Wagon",
            "Gigabyte_GAZ97XSLI_10_motherboard_ATX_LGA1150_Socket_Z97",
            "Gigabyte_GA970AUD3P_10_Motherboard_ATX_Socket_AM3",
            "Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3",
            "2_of_Jenga_Classic_Game", "50_BLOCKS", "5_HTP",
            "ASICS_GELAce_Pro_Pearl_WhitePink", "ASICS_GELBlur33_20_GS_BlackWhiteSafety_Orange",
            "Asus_M5A78LMUSB3_Motherboard_Micro_ATX_Socket_AM3",
            "Asus_M5A99FX_PRO_R20_Motherboard_ATX_Socket_AM3",
            "Asus_Sabertooth_990FX_20_Motherboard_ATX_Socket_AM3",
            "Asus_Sabertooth_Z97_MARK_1_Motherboard_ATX_LGA1150_Socket",
            "BlackBlack_Nintendo_3DSXL", "Black_Elderberry_Syrup_54_oz_Gaia_Herbs",
            "BlueBlack_Nintendo_3DSXL", "Blue_Jasmine_Includes_Digital_Copy_UltraViolet_DVD",
            "30_CONSTRUCTION_SET", "3D_Dollhouse_Swing",
            "3M_Antislip_Surfacing_Light_Duty_White", "3M_Vinyl_Tape_Green_1_x_36_yd",
        ]
    
    def get_models_from_fuel_api(self) -> List[str]:
        """Obtém lista de modelos via API do Fuel com paginação."""
        models = []
        page = 1
        per_page = 100
        
        print("[INFO] Obtendo lista de modelos via API do Fuel...")
        
        while len(models) < self.max_objects:
            try:
                endpoint = f"{self.FUEL_API}/1.0/models"
                params = {
                    'owner': self.GOOGLE_RESEARCH_OWNER,
                    'page': page,
                    'per_page': per_page
                }
                
                response = self.session.get(endpoint, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    page_models = []
                    if isinstance(data, list):
                        page_models = [item.get('name') for item in data if isinstance(item, dict) and 'name' in item]
                    elif isinstance(data, dict):
                        if 'data' in data:
                            page_models = [item.get('name') for item in data['data'] if isinstance(item, dict) and 'name' in item]
                        elif 'models' in data:
                            page_models = [item.get('name') for item in data['models'] if isinstance(item, dict) and 'name' in item]
                    
                    if not page_models:
                        break
                    
                    models.extend([m for m in page_models if m and m not in models])
                    print(f"[INFO] Página {page}: {len(page_models)} modelos, total: {len(models)}")
                    
                    if len(page_models) < per_page or len(models) >= self.max_objects:
                        break
                    
                    page += 1
                    time.sleep(0.5)
                else:
                    break
                    
            except Exception as e:
                print(f"[AVISO] Erro na página {page}: {e}")
                break
        
        if models:
            print(f"[OK] Total de {len(models)} modelos encontrados via API")
        
        return models
    
    def get_model_list(self) -> List[str]:
        """Obtém lista completa de modelos."""
        # Usar lista conhecida diretamente (já tem 200+ modelos)
        models_list = self.known_models[:self.max_objects]
        print(f"[OK] {len(models_list)} modelos da lista para baixar")
        return models_list
    
    def _scrape_models(self) -> List[str]:
        """Scraping básico da página."""
        models = []
        try:
            # Acessar página do owner diretamente
            url = f"{self.BASE_URL}/{self.GOOGLE_RESEARCH_OWNER}/models"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Extrair nomes de modelos
                pattern = r'/GoogleResearch/models/([^"\'<>/\s?&]+)'
                matches = re.findall(pattern, response.text)
                models = [m for m in matches if m and len(m) > 1]
        except:
            pass
        
        return models
    
    def _download_file_from_tree(self, model_name: str, file_path: str, model_dir: Path) -> bool:
        """Baixa um arquivo individual da árvore de arquivos."""
        try:
            model_url_encoded = quote(model_name, safe='')
            file_url_encoded = quote(file_path.lstrip('/'), safe='')
            
            # URL para baixar arquivo individual
            file_url = f"{self.FUEL_API}/1.0/{self.GOOGLE_RESEARCH_OWNER}/models/{model_url_encoded}/tip/files{file_path}"
            
            response = self.session.get(file_url, timeout=30, stream=True)
            
            if response.status_code == 200:
                # Criar estrutura de diretórios
                local_path = model_dir / file_path.lstrip('/')
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Salvar arquivo
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return True
        except Exception as e:
            pass
        return False
    
    def _download_file_tree(self, model_name: str, file_tree: list, model_dir: Path) -> int:
        """Baixa recursivamente todos os arquivos da árvore."""
        downloaded = 0
        
        def process_node(node):
            nonlocal downloaded
            if 'children' in node:
                # É um diretório, processar filhos
                for child in node['children']:
                    process_node(child)
            else:
                # É um arquivo, baixar
                file_path = node.get('path', '')
                if file_path and self._download_file_from_tree(model_name, file_path, model_dir):
                    downloaded += 1
        
        for node in file_tree:
            process_node(node)
        
        return downloaded
    
    def download_model(self, model_name: str) -> bool:
        """Baixa um modelo completo."""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        print(f"[INFO] Baixando: {model_name}")
        
        try:
            model_url_encoded = quote(model_name, safe='')
            
            # 1. Obter informações do modelo
            model_info_url = f"{self.FUEL_API}/1.0/{self.GOOGLE_RESEARCH_OWNER}/models/{model_url_encoded}"
            model_info = None
            try:
                info_response = self.session.get(model_info_url, timeout=30)
                if info_response.status_code == 200:
                    model_info = info_response.json()
                    # Salvar metadados
                    with open(self.metadata_dir / f"{model_name}.json", 'w', encoding='utf-8') as f:
                        json.dump(model_info, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[AVISO] Erro ao obter metadados: {e}")
            
            # 2. Obter file_tree
            files_url = f"{self.FUEL_API}/1.0/{self.GOOGLE_RESEARCH_OWNER}/models/{model_url_encoded}/tip/files"
            try:
                files_response = self.session.get(files_url, timeout=30)
                if files_response.status_code == 200:
                    files_data = files_response.json()
                    file_tree = files_data.get('file_tree', [])
                    
                    if file_tree:
                        # Baixar todos os arquivos da árvore
                        downloaded_count = self._download_file_tree(model_name, file_tree, model_dir)
                        
                        if downloaded_count > 0:
                            # Extrair imagens
                            self._extract_images(model_dir, model_name)
                            
                            print(f"[OK] {model_name} baixado ({downloaded_count} arquivos)")
                            return True
                        else:
                            print(f"[AVISO] Nenhum arquivo baixado para {model_name}")
                    else:
                        print(f"[AVISO] File tree vazio para {model_name}")
                else:
                    print(f"[AVISO] Erro ao obter file_tree: {files_response.status_code}")
            except Exception as e:
                print(f"[AVISO] Erro ao processar file_tree: {e}")
            
            return False
            
            response = self.session.get(api_url, stream=True, timeout=60, allow_redirects=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                
                # Verificar se é JSON (lista de arquivos)
                if 'application/json' in content_type or response.text.strip().startswith('{') or response.text.strip().startswith('['):
                    try:
                        files_data = response.json()
                        
                        # Processar lista de arquivos
                        if isinstance(files_data, list):
                            files_list = files_data
                        elif isinstance(files_data, dict):
                            files_list = files_data.get('files', files_data.get('data', []))
                        else:
                            files_list = []
                        
                        # Baixar cada arquivo
                        downloaded_files = []
                        for file_info in files_list:
                            if isinstance(file_info, dict):
                                file_url = file_info.get('url') or file_info.get('download_url') or file_info.get('file_path')
                                file_name = file_info.get('name') or file_info.get('file_name') or Path(file_url).name
                                
                                if file_url:
                                    if not file_url.startswith('http'):
                                        file_url = urljoin(self.FUEL_API, file_url)
                                    
                                    try:
                                        file_response = self.session.get(file_url, timeout=30)
                                        if file_response.status_code == 200:
                                            file_path = model_dir / file_name
                                            file_path.parent.mkdir(parents=True, exist_ok=True)
                                            with open(file_path, 'wb') as f:
                                                f.write(file_response.content)
                                            downloaded_files.append(file_name)
                                    except:
                                        pass
                        
                        if downloaded_files:
                            # Extrair imagens
                            self._extract_images(model_dir, model_name)
                            
                            # Salvar metadados
                            metadata = {
                                'name': model_name,
                                'owner': self.GOOGLE_RESEARCH_OWNER,
                                'files': downloaded_files,
                                'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
                            }
                            with open(self.metadata_dir / f"{model_name}.json", 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, indent=2)
                            
                            print(f"[OK] {model_name} baixado ({len(downloaded_files)} arquivos)")
                            return True
                    except json.JSONDecodeError:
                        pass
                
                # Tentar como ZIP
                zip_path = model_dir / f"{model_name}.zip"
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Verificar se é ZIP válido
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(model_dir)
                    zip_path.unlink()
                    
                    # Extrair imagens
                    self._extract_images(model_dir, model_name)
                    
                    # Salvar metadados
                    metadata = {
                        'name': model_name,
                        'owner': self.GOOGLE_RESEARCH_OWNER,
                        'download_url': api_url,
                        'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    with open(self.metadata_dir / f"{model_name}.json", 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                    
                    print(f"[OK] {model_name} baixado com sucesso")
                    return True
                except zipfile.BadZipFile:
                    # Não é ZIP, salvar como está
                    zip_path.rename(model_dir / f"{model_name}_raw.bin")
                    print(f"[AVISO] Arquivo não é ZIP, salvo como raw para {model_name}")
            
            # 2. Fallback: baixar página e extrair informações
            page_url = f"{self.BASE_URL}/{self.GOOGLE_RESEARCH_OWNER}/models/{model_url_encoded}"
            page_response = self.session.get(page_url, timeout=30)
            
            if page_response.status_code == 200:
                # Salvar HTML
                html_path = model_dir / "page.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(page_response.text)
                
                # Extrair imagens da página
                img_patterns = [
                    r'src="([^"]+\.(jpg|jpeg|png|gif))"',
                    r'href="([^"]+\.(jpg|jpeg|png|gif))"',
                ]
                
                for pattern in img_patterns:
                    matches = re.findall(pattern, page_response.text, re.IGNORECASE)
                    for match in matches[:3]:  # Limitar a 3 imagens
                        img_url = match[0] if isinstance(match, tuple) else match
                        if not img_url.startswith('http'):
                            img_url = urljoin(page_url, img_url)
                        self._download_image(img_url, model_name)
                
                print(f"[OK] {model_name} processado (modo fallback)")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERRO] Erro ao baixar {model_name}: {e}")
            return False
    
    def _extract_images(self, model_dir: Path, model_name: str):
        """Extrai imagens do diretório do modelo."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        for ext in image_extensions:
            for img_file in model_dir.rglob(f'*{ext}'):
                if img_file.is_file():
                    dest = self.images_dir / f"{model_name}_{img_file.name}"
                    try:
                        shutil.copy2(img_file, dest)
                    except:
                        pass
    
    def _download_image(self, img_url: str, model_name: str):
        """Baixa uma imagem individual."""
        try:
            response = self.session.get(img_url, timeout=30)
            if response.status_code == 200:
                ext = Path(urlparse(img_url).path).suffix or '.jpg'
                img_path = self.images_dir / f"{model_name}{ext}"
                with open(img_path, 'wb') as f:
                    f.write(response.content)
        except:
            pass
    
    def download_dataset(self):
        """Baixa o dataset completo."""
        print("=" * 60)
        print("[INFO] Iniciando download do dataset GoogleResearch")
        print(f"[INFO] Objetivo: {self.max_objects} objetos")
        print("=" * 60)
        
        models = self.get_model_list()
        
        if not models:
            print("[ERRO] Nenhum modelo encontrado!")
            return
        
        print(f"[OK] {len(models)} modelos para processar\n")
        
        for i, model_name in enumerate(models, 1):
            print(f"[{i}/{len(models)}] ", end="")
            
            if self.download_model(model_name):
                self.downloaded_count += 1
            else:
                self.failed_count += 1
            
            time.sleep(0.5)  # Rate limiting
            
            if i % 10 == 0:
                self._save_progress(models, i)
        
        print("\n" + "=" * 60)
        print("[RESUMO] Download concluído!")
        print("=" * 60)
        print(f"Baixados: {self.downloaded_count}")
        print(f"Falhas: {self.failed_count}")
        print(f"Total: {len(models)}")
        print(f"\nDataset em: {self.output_dir.absolute()}")
    
    def _save_progress(self, models: List[str], current_index: int):
        """Salva progresso."""
        progress_file = self.output_dir / "download_progress.json"
        progress = {
            'total': len(models),
            'processed': current_index,
            'downloaded': self.downloaded_count,
            'failed': self.failed_count,
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Baixa dataset do Google Research do Gazebo")
    parser.add_argument('--output', default='data/raw/gazebo_dataset', help='Diretório de saída')
    parser.add_argument('--max-objects', type=int, default=200, help='Número máximo de objetos')
    
    args = parser.parse_args()
    
    downloader = GazeboDatasetDownloaderV2(
        output_dir=args.output,
        max_objects=args.max_objects
    )
    
    downloader.download_dataset()


if __name__ == '__main__':
    main()

