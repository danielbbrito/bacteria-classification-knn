import matplotlib.pyplot as plt
import cv2 as cv
import math
import os


def letterbox(img):
    # Retorna (height, width, channels)
    height, width = img.shape[:2]
    size = max(height, width)
    
    top = math.ceil((size - height) / 2)
    bottom = math.floor((size - height) / 2)
    left = math.ceil((size - width) / 2)
    right = math.floor((size - width) / 2)

    return cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0, 0, 0))




def main():
    # Preprocessar imagens parasitadas
    dir = "imagens_celulas/Parasitized"

    img_parasitized = []
    print("Carregando imagens infectadas")
    
    for entry in os.listdir(dir):
        if entry == "Thumbs.db": 
            continue
        
        file_path = os.path.join(dir, entry)
        img = cv.imread(file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        

        img_cpy = img
        img = letterbox(img) # padronizamos a geometria da imagem para o resizing não distorce-la

        if entry == "C37BP2_thinF_IMG_20150620_133111a_cell_87.png":
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title("a")
            ax[0].imshow(img_cpy)
         
            ax[1].set_title("b")
            ax[1].imshow(img)

            plt.savefig("figures/letterboxed_parasitised.png")
            plt.close(fig)
        
        img_cpy = img
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA)

        if entry == "C37BP2_thinF_IMG_20150620_133111a_cell_87.png":
            
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title("c")
            ax[0].imshow(img_cpy)
          
            ax[1].set_title("d")
            ax[1].imshow(img)

            plt.savefig("figures/resized_parasitised.png")
            plt.close(fig)

        # Vamos colocar em grayscale
        img_cpy = img
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        if entry == "C37BP2_thinF_IMG_20150620_133111a_cell_87.png":
            
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title("e")
            ax[0].imshow(img_cpy)
          
            ax[1].set_title("f")
            ax[1].imshow(img, cmap="gray") # Usar o colormap gray é necessário, se não fica verde!

            plt.savefig("figures/grayscale_parasitised.png")
            plt.close(fig)

            plt.hist(img.flatten(), bins=256, range=(0,256))
            plt.yscale("log")
            plt.xlabel("Intensidade da cor do Pixel")
            plt.ylabel("Frequência (log)")
            plt.savefig("figures/color_hist.png") # Plotamos o histograma de cores antes da equalização
            plt.close()
          

        # Agora aplicamos CLAHE para normalizar o histograma da imagem
        img_cpy = img
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
        img = clahe.apply(img)

        if entry == "C37BP2_thinF_IMG_20150620_133111a_cell_87.png":
            
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title("g")
            ax[0].imshow(img_cpy, cmap="gray")
          
            ax[1].set_title("h")
            ax[1].imshow(img, cmap="gray")

            plt.savefig("figures/contrast_enhanced_parasitised.png")
            plt.close(fig)

            plt.hist(img.flatten(), bins=256, range=(0,256),color="#ff0000")
            plt.yscale("log")
            plt.xlabel("Intensidade da cor do Pixel")
            plt.ylabel("Frequência (log)")
            plt.savefig("figures/color_hist_enhanced.png") # Plotamos o histograma de cores também depois
            plt.close()

        img_parasitized.append(img)

    print("Imagens infectadas carregadas")
    print()
    
    dir = "imagens_celulas/Uninfected"
    img_uninfected = []

    # Faremos o mesmo para as não infectadas
    print("Carregando imagens nã infectadas")
    for entry in os.listdir(dir):
        if entry == "Thumbs.db": 
            continue
        
        file_path = os.path.join(dir, entry)
        img = cv.imread(file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img_cpy = img
        img = letterbox(img)

        img_cpy = img
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA)

        img_cpy = img
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        img_cpy = img
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
        img = clahe.apply(img)

        img_uninfected.append(img)
    
   
    print("Imagens não infectadas carregadas")
    print()

    

    

if __name__ == "__main__":
    main()