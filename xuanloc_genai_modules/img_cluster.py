import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from xuanloc_utils import common
from sentence_transformers import SentenceTransformer
import faiss

class ImgCluster:
    def __init__(self, top_k=50, dis=0.01):
        self.model = SentenceTransformer('clip-ViT-B-32') # clip-ViT-L-14, clip-ViT-B-16, clip-ViT-B-32
        self.top_k = top_k
        self.dis = dis
        
    def init_embeddings(self, input_path):
        embeddings_path = os.path.basename(input_path) + '.npy'  

        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
            print(f'Load embeddings from {embeddings_path}')
        else:
            embeddings = []
            _, img_paths = common.get_items_from_folder(input_path, exts=['jpg', 'png'])
            for img_path in tqdm(img_paths, desc='Creating embeddings'):
                img = Image.open(img_path)
                img_embedding = self.model.encode(img)
                embeddings.append(img_embedding)
                
            np.save(embeddings_path, embeddings)
            print(f'Save embeddings to {embeddings_path}')

        return embeddings
    
    def init_index(self, input_path, embeddings):
        faiss_path = os.path.basename(input_path) + '.faiss'
        if os.path.exists(faiss_path):
            index = faiss.read_index(faiss_path)
            print(f'Load index from {faiss_path}')
        else:   
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIDMap(index)

            vectors = np.array(embeddings).astype('float32')
            index.add_with_ids(vectors, np.array(range(len(embeddings))))

            # Save index 
            faiss.write_index(index, faiss_path)
            print(f'Save index to {faiss_path}')

        return index
    
    def build(self, input_path):
        self.embeddings = self.init_embeddings(input_path)
        self.index = self.init_index(input_path, self.embeddings)
    
    def remove_overlap(self, input_path, output_path, is_save_overlap=False):
        # build
        self.build(input_path)
        
        common.create_folder(output_path)
        if is_save_overlap:
            overlap_path = os.path.join(output_path, 'overlap')
            common.create_folder(overlap_path)
        img_names, img_paths = common.get_items_from_folder(input_path, exts=['jpg', 'png'])
        
        # ///////////////////////////////////////////////////
        overlap_img_names_list = []
        for i, (img_name, img_path, embedding) in tqdm(enumerate(zip(img_names, img_paths, self.embeddings)), desc='Search similar images', total=len(img_names)):
            # check if img_name is in overlap_img_names_list
            is_in = False
            for overlap_img_names in overlap_img_names_list:
                if img_name in overlap_img_names:
                    is_in = True
                    break
            if is_in:
                continue
                
            # search similar images
            dis_list, idxs = self.index.search(embedding.reshape(1, -1), self.top_k)
            
            # get similar images
            similar_img_names = []
            for dis, idx in zip(dis_list[0], idxs[0]):
                if dis < self.dis:
                    similar_img_names.append(img_names[idx])
            overlap_img_names_list.append(similar_img_names)

        # ///////////////////////////////////////////////////
        stats = {
            'total': len(img_names),
            'overlap': 0,
            'overlap_unique': 0,
            'unique': 0,
        }
        for i, overlap_img_names in tqdm(enumerate(overlap_img_names_list), desc='Remove overlap images', total=len(overlap_img_names_list)):
            img_name = overlap_img_names[0]
            img_path = img_paths[img_names.index(img_name)]
            new_img_path = os.path.join(output_path, img_name)
            shutil.copy(img_path, new_img_path)
            
            if len(overlap_img_names) > 1:
                if is_save_overlap:
                    folder_name = f'len({len(overlap_img_names)})_{i}'
                    folder_path = os.path.join(overlap_path, folder_name)
                    common.create_folder(folder_path)
                    
                    for img_name in overlap_img_names:
                        img_path = img_paths[img_names.index(img_name)]
                        new_img_path = os.path.join(folder_path, img_name)
                        shutil.copy(img_path, new_img_path)
                    
                stats['overlap'] += len(overlap_img_names)
                stats['overlap_unique'] += 1
                
            elif len(overlap_img_names) == 1:
                stats['unique'] += 1
                
        print(stats)

if __name__ == '__main__':
    cluster = ImgCluster()
    cluster.remove_overlap('uniform_data/prison', 'uniform_data_not_overlap/prison', is_save_overlap=False)
    cluster.remove_overlap('uniform_data/other', 'uniform_data_not_overlap/other', is_save_overlap=False)
    cluster.remove_overlap('uniform_data/police', 'uniform_data_not_overlap/police', is_save_overlap=False)
    cluster.remove_overlap('uniform_data', 'out', is_save_overlap=False)
    
    img_names, img_paths = common.get_items_from_folder('uniform_data', exts=['jpg', 'png'])
    print(len(img_names))
    
    img_names, img_paths = common.get_items_from_folder('out', exts=['jpg', 'png'])
    print(len(img_names))
    
    img_names, img_paths = common.get_items_from_folder('uniform_data_not_overlap', exts=['jpg', 'png'])
    print(len(img_names))