#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download direto de modelos usando lista de nomes."""

import json
import time
import requests
from pathlib import Path
from urllib.parse import quote
import shutil

# Lista de modelos fornecida
MODEL_NAMES = [
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
    "Gigabyte_GA78LMTUSB3_50_Motherboard_Micro_ATX_Socket_AM3"
]


class DirectDownloader:
    """Download direto usando lista de nomes."""
    
    FUEL_API = "https://fuel.gazebosim.org"
    OWNER = "GoogleResearch"
    
    def __init__(self, output_dir: str = "data/raw/gazebo_dataset", model_list: list = None):
        self.output_dir = Path(output_dir)
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
        
        self.model_list = model_list or MODEL_NAMES[:200]
        self.downloaded = 0
        self.failed = 0
    
    def download_file(self, model_name: str, file_path: str, model_dir: Path) -> bool:
        """Baixa um arquivo individual."""
        try:
            model_encoded = quote(model_name, safe='')
            file_url = f"{self.FUEL_API}/1.0/{self.OWNER}/models/{model_encoded}/tip/files{file_path}"
            
            response = self.session.get(file_url, timeout=30, stream=True)
            if response.status_code == 200:
                local_path = model_dir / file_path.lstrip('/')
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Extrair imagens
                if file_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    dest = self.images_dir / f"{model_name}_{Path(file_path).name}"
                    shutil.copy2(local_path, dest)
                
                return True
        except:
            pass
        return False
    
    def download_model(self, model_name: str) -> bool:
        """Baixa um modelo completo."""
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        try:
            model_encoded = quote(model_name, safe='')
            
            # Metadados
            info_url = f"{self.FUEL_API}/1.0/{self.OWNER}/models/{model_encoded}"
            try:
                info_resp = self.session.get(info_url, timeout=30)
                if info_resp.status_code == 200:
                    with open(self.metadata_dir / f"{model_name}.json", 'w', encoding='utf-8') as f:
                        json.dump(info_resp.json(), f, indent=2, ensure_ascii=False)
            except:
                pass
            
            # File tree
            files_url = f"{self.FUEL_API}/1.0/{self.OWNER}/models/{model_encoded}/tip/files"
            files_resp = self.session.get(files_url, timeout=30)
            
            if files_resp.status_code != 200:
                return False
            
            file_tree = files_resp.json().get('file_tree', [])
            if not file_tree:
                return False
            
            # Baixar arquivos
            downloaded = 0
            
            def process_node(node):
                nonlocal downloaded
                if 'children' in node:
                    for child in node['children']:
                        process_node(child)
                else:
                    file_path = node.get('path', '')
                    if file_path and self.download_file(model_name, file_path, model_dir):
                        downloaded += 1
            
            for node in file_tree:
                process_node(node)
            
            return downloaded > 0
            
        except Exception as e:
            return False
    
    def download_all(self):
        """Baixa todos os modelos da lista."""
        print("=" * 60)
        print(f"[INFO] Download Direto - {len(self.model_list)} modelos")
        print("=" * 60)
        
        for i, model_name in enumerate(self.model_list, 1):
            print(f"[{i}/{len(self.model_list)}] ", end="", flush=True)
            
            if self.download_model(model_name):
                self.downloaded += 1
                print(f"[OK] {model_name}")
            else:
                self.failed += 1
                print(f"[FALHOU] {model_name}")
            
            time.sleep(0.3)
            
            if i % 10 == 0:
                self._save_progress(i)
        
        print("\n" + "=" * 60)
        print(f"[RESUMO] Baixados: {self.downloaded}/{len(self.model_list)}")
        print(f"[RESUMO] Falhas: {self.failed}")
        print(f"[RESUMO] Dataset em: {self.output_dir.absolute()}")
    
    def _save_progress(self, current: int):
        """Salva progresso."""
        progress = {
            'total': len(self.model_list),
            'processed': current,
            'downloaded': self.downloaded,
            'failed': self.failed
        }
        with open(self.output_dir / "download_progress.json", 'w') as f:
            json.dump(progress, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download direto de modelos Gazebo")
    parser.add_argument('--max-models', type=int, default=200, help='Número máximo')
    parser.add_argument('--output', default='data/raw/gazebo_dataset', help='Diretório de saída')
    
    args = parser.parse_args()
    
    model_list = MODEL_NAMES[:args.max_models]
    downloader = DirectDownloader(output_dir=args.output, model_list=model_list)
    downloader.download_all()


if __name__ == '__main__':
    main()


