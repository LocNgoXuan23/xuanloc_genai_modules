import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from xuanloc_utils import common
from sentence_transformers import SentenceTransformer
import faiss

class ImgSearcher:
    def __init__(self, top_k=5):
        self.model = SentenceTransformer('clip-ViT-B-32') # clip-ViT-L-14, clip-ViT-B-16, clip-ViT-B-32
        self.top_k = top_k

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
            index = faiss.IndexFlatIP(dimension)
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

    def search(self, query):
        if query.endswith(".jpg"):
            query = Image.open(query)

        query_embedding = self.model.encode(query)
        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        dis_list, idxs = self.index.search(query_embedding, self.top_k)
        print(dis_list)
        return dis_list, idxs
    
if __name__ == '__main__':
    input_path = 'uniform_data'
    output_path = 'out'
    common.create_folder(output_path)
    img_names, img_paths = common.get_items_from_folder(input_path, exts=['jpg', 'png'])
    
    searcher = ImgSearcher()
    searcher.build('uniform_data')
    dis_list, idxs = searcher.search('vietnam prison_000001_380_96_899_740.jpg')
    print(dis_list)
    print(idxs)
    
    similar_img_paths = [img_paths[i] for i in idxs[0]]
    for i, similar_img_path in enumerate(similar_img_paths):
        similar_img = Image.open(similar_img_path)
        similar_img.save(os.path.join(output_path, f"{i}.jpg"))