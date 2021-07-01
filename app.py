import sys
from flask import Flask, render_template, request, jsonify

import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

################################
#Carrega modelos
################################

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

def numeric_to_string(y):
  if(y == 0):
    return 'CNV'
  elif(y == 1):
    return 'DME'
  elif(y == 2):
    return 'DRUSEN'
  elif(y == 3):
    return 'NORMAL'

class PerturbationCAM():

  #Passa a rede extratora de features, o modelo após a extração das features e o shape de input da rede inicial
  def __init__(self, featurenet, model, eh_nn = False, shape_input = (512, 512)):
    self.featurenet = featurenet
    self.model = model
    self.eh_nn = eh_nn
    self.shape_input = shape_input

  #Faz a predição de um batch de imagens (replica em 3 canais, extrai as features e faz o predict)
  #Note que se for uma rede neural o predict não é o mesmo método que um classificador com predict_proba
  def predict(self, batch_img):
    batch_img = np.repeat(batch_img[:, :, :, np.newaxis], 3, -1) #Faz virar imagens de 3 canais
    features_ref = self.featurenet.predict(batch_img)
    if(self.eh_nn):
      return self.model.predict(features_ref)
    else:
      return self.model.predict_proba(features_ref)

  #Faz a predição de uma só imagem e já retorna a classe predita em string
  def predict_label_one_img(self, img_array):
    batch_img = np.array([img_array])
    batch_img = np.repeat(batch_img[:, :, :, np.newaxis], 3, -1) #Faz virar imagens de 3 canais
    features_ref = self.featurenet.predict(batch_img)
    if(self.eh_nn):
      return numeric_to_string(np.argmax(self.model.predict(features_ref)[0]))
    else:
      return self.model.predict(features_ref)[0][0]
  
  #Faz os tratamentos necessários para a imagem de input 
  #(por enquanto só reescala para o shape da rede de extração de features e normaliza)
  def trata_imagem(self, img_array):
    return cv2.resize(img_array, dsize = self.shape_input[::-1])/255.0

  #Recebe uma imagem só com um canal (preto e branco) normalizado entre 0 e 1
  #Retorna o mapa de calor para interpretabilidade
  def cria_mapa_calor(self, img_array, batch_cam = 32, num_random_mask = 10):
    shape_original = img_array.shape

    img_array = self.trata_imagem(img_array)

    #Faz uma predição da imagem sem alterações
    pred_ref = self.predict(np.array([img_array]))[0]

    #Faz um scan da imagem em conjunto de pixels dado pelo batch_cam
    i_max = int(img_array.shape[0]/batch_cam)
    j_max = int(img_array.shape[1]/batch_cam)
    perturbation_matrix = np.empty(shape = (i_max, j_max))
    for i in range(0, i_max):
      for j in range(0, j_max):
        #Para cada etapa do scan, alteramos os valores desses pixels da imagens nessa região num_random_mask vezes
        #E verificamos como essa alteração altera o valor predito
        batch = []
        for k in range(0, num_random_mask):
          img_temp = img_array.copy()
          img_temp[(batch_cam*i):(batch_cam*(i+1)), (batch_cam*j):(batch_cam*(j+1))] = np.random.rand(batch_cam, batch_cam)
          batch.append(img_temp)
        batch = np.array(batch)
        #Definimos a importância desse conjunto de pixels como a intensidade da perturbação média ocorrida na predição
        perturbation_matrix[i, j] = np.sum(np.sum((self.predict(batch) - pred_ref)**2, axis = 1)**0.5)/num_random_mask
    #Normaliza o valor das perturbações entre 0 e 1 (será nosso mapa de calor)
    perturbation_matrix = perturbation_matrix/np.max(perturbation_matrix)
    
    #Retornamos reescalando o mapa de calor para o shape da imagem original
    return cv2.resize(perturbation_matrix, dsize = shape_original[::-1])

#Dá cor para nosso mapa de calor
#Se use_heatmap_alpha = False retorna um mapa de calor que vai do azul para o vermelho
#Se use_heatmap_alpha = True retorna um mapa sempre vermelho mas com variação de alpha
def colorir_heatmap(img_heatmap, use_heatmap_alpha, alpha = 0.4):
    if(use_heatmap_alpha == False):
      jet = cm.get_cmap("jet") #Objeto para colorir o heatmap
      jet_colors = jet(np.arange(256))[:, :3]
      heatmap = np.uint8(255 * img_heatmap) # Reescala o heatmap entre 0 e 255 (inteiro)
      jet_heatmap = jet_colors[heatmap]

      #Define um alpha constante
      heatmap_alpha = img_heatmap
      jet_heatmap_nova = np.empty(shape = (img_heatmap.shape[0], img_heatmap.shape[1], 4))
      jet_heatmap_nova[:, :, :3] = jet_heatmap
      jet_heatmap_nova[:, :, 3] = alpha
      jet_heatmap = jet_heatmap_nova

    else:
      N = 256
      vals = np.ones((N, 4))
      vals[:, 0] = 1.0
      vals[:, 1] = 0.0
      vals[:, 2] = 0.0
      newcmp = ListedColormap(vals)
      jet_colors = newcmp(np.arange(256))[:, :3]
      heatmap = np.uint8(255 * img_heatmap) # Reescala o heatmap entre 0 e 255 (inteiro)
      jet_heatmap = jet_colors[heatmap]

      #Usa a própria intensidade da perturbação como alpha
      heatmap_alpha = img_heatmap*alpha
      jet_heatmap_nova = np.empty(shape = (img_heatmap.shape[0], img_heatmap.shape[1], 4))
      jet_heatmap_nova[:, :, :3] = jet_heatmap
      jet_heatmap_nova[:, :, 3] = heatmap_alpha
      jet_heatmap = jet_heatmap_nova

    return (jet_heatmap * 255).astype(np.uint8)

path_models = 'modelos/'


with open(path_models + 'featurenet.json', 'r') as json_file:
    featurenet_json = json_file.read()
featurenet = tf.keras.models.model_from_json(featurenet_json)
featurenet.load_weights(path_models + 'featurenet.h5')

with open(path_models + 'model_baseline_nn.json', 'r') as json_file:
    model_baseline_nn_json = json_file.read()
model_baseline_nn = tf.keras.models.model_from_json(model_baseline_nn_json)
model_baseline_nn.load_weights(path_models + 'weights_baseline.h5')

pert_cam = PerturbationCAM(featurenet, model_baseline_nn, eh_nn = True) 
#pert_cam = PerturbationCAM(featurenet, clf_cat, eh_nn = False) 

################################
#
################################

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    if request.method == 'POST':
        file_img = request.files['file']
        
        #Converte imagem para o encoder base64 (fácil para enviar para o HTML como string)
        img_base64 = base64.b64encode(file_img.read())
        img_base64_string_html = 'data:image/jpeg;base64,' + str(img_base64)[2:-1]
        
        #print(img_base64_string[:50], file = sys.stderr)
        
        #lê imagem e converte para grayscale se necessário
        img_array = np.array(Image.open(io.BytesIO(base64.b64decode(img_base64))))
        if(len(img_array.shape) == 3 and img_array.shape[2] == 3):
            img_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        
        probs = [float("%.2f" % p) for p in pert_cam.predict(np.array([pert_cam.trata_imagem(img_array)]))[0]]
        predito = pert_cam.predict_label_one_img(pert_cam.trata_imagem(img_array))
        
        #print(predito, file = sys.stderr)
        
        img_heatmap = pert_cam.cria_mapa_calor(img_array, batch_cam = 64, num_random_mask = 5)
        
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        ax[0].imshow(img_array, cmap = 'gray')
        ax[0].set_title('Imagem:')
        ax[0].set_axis_off()
        
        ax[1].imshow(img_heatmap, cmap = cm.get_cmap("jet"))
        ax[1].set_title('Probs: ' + str(probs))
        ax[1].set_axis_off()

        heatmap_colorido = colorir_heatmap(img_heatmap, use_heatmap_alpha = True, alpha = 0.75)
        ax[2].imshow(img_array, cmap = 'gray')
        ax[2].imshow(heatmap_colorido)
        ax[2].set_title('Predição: ' + predito)
        ax[2].set_axis_off()
        
        #Converte a imagem do matplotlib para o encoder base64 para enviar para o HTML
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, format = 'jpg')
        my_stringIObytes.seek(0)
        my_base64_jpgData = base64.b64encode(my_stringIObytes.read())
        final_img_base64_string_html = 'data:image/jpeg;base64,' + str(my_base64_jpgData)[2:-1]
        
        return jsonify(result = final_img_base64_string_html)

@app.route("/")
def index():
    # Load current count
    f = open("count.txt", "r")
    count = int(f.read())
    f.close()

    # Increment the count
    count += 1

    # Overwrite the count
    f = open("count.txt", "w")
    f.write(str(count))
    f.close()

    # Render HTML with count variable
    return render_template("index.html", count = count)

if __name__ == "__main__":
    app.run()