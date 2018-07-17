import csv
import sys
import requests
import skimage.io
import os
import os.path

import glob
import pickle
import time

from IPython.display import display, Image, HTML
from keras.applications import VGG16
import datetime

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image as kimage
import numpy as np
import pandas as pd
import scipy.sparse as sp

import skimage.io
import hashlib


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.python.client import device_lib

import tensorflow as tf


import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, Image as dImage, HTML
from shutil import copyfile


def calcula_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_arquivos_hashs(fonte, arqs):
    raiz = arqs[fonte]["raiz"]
    print("\n  Calcula hashs dos arquivos", fonte)
    print("  Diretório raiz", raiz)
    num = 0
    for r,d,f in os.walk(raiz):
        for arquivo_rel in f:
            arquivo_abs = os.path.join(r, arquivo_rel)
            arquivo_rel = arquivo_abs[len(raiz):]
            ind = arquivo_rel.rfind(".")
            #num+=1
            #if (num%500==0): 
            #    z=input("digite algo: ")
            #    if z=="q": return
            if ind>0 and os.path.isfile(arquivo_abs):
                #print(arquivo_abs, ext)
                ext = arquivo_rel[ind:].lower()
                if ext in arqs["extensoes_img"]:
                    md5 = calcula_md5(arquivo_abs)
                    try:
                        img = skimage.io.imread(arquivo_abs)
                        arqs[fonte]["nomes"].append(arquivo_rel)
                        if md5 not in arqs[fonte]["hashs"]:
                            arqs[fonte]["hashs"][md5] = []
                        arqs[fonte]["hashs"][md5].append(arquivo_rel)
                        num+=1
                        if (num%500==0): print("    -->",num)
                    except:
                        print("  Erro arquivo de imagem:", arquivo_abs)
    print("    -->",num)


def grava_arqs(arqs):
    filehandler = open("arqs.pickle", 'wb') 
    pickle.dump(arqs, filehandler)


def le_arqs():
    filehandler=open("arqs.pickle", 'rb') 
    ret = pickle.load(filehandler)
    return ret


# estrutura de arqs
def print_arqs(arqs):
    for s in arqs.keys():
        print("\n  arqs[\"" + s + "\"]")
        if 'dict' in str(type(arqs[s])):
            for t in arqs[s].keys():
                print("    arqs[\"" + s + "\"][\"" + t + "\"]  ---  tipo:", type(arqs[s][t]))
        

# estrutura de dados
def print_dados(dados):
    for s in dados.keys():
        print("\n  dados[\"" + s + "\"]")
        if 'dict' in str(type(dados[s])):
            for t in dados[s].keys():
                print("    dados[\"" + s + "\"][\"" + t + "\"]  ---  tipo:", type(dados[s][t]))


def grava_dados_pred(dados):
    filehandler = open("dados_pred.pickle", 'wb') 
    pickle.dump(dados, filehandler)

def le_dados_pred():
    filehandler = open("dados_pred.pickle", 'rb') 
    ret = pickle.load(filehandler)
    return ret


def get_available_devices():  
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


def comp_similarity(ratings):
    print("  similaridade")
    sim = ratings.dot(ratings.T)
    if not isinstance(sim, np.ndarray):
        sim = sim.toarray()
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)


def comp_similarity_tf(ratings):
    print("  similaridade tensor flow")
    sess = tf.Session()
    
    matrix1 = tf.convert_to_tensor(ratings, np.float32)
    matrix2 = tf.transpose(matrix1)
    sim = tf.matmul(matrix1, matrix2)
    diag0 = tf.diag_part(sim)
    diag = tf.zeros([1, diag0.get_shape()[0]], tf.float32)
    diag = tf.add(diag, diag0)
    norms = tf.sqrt(diag)
    normst = tf.transpose(norms)
    div1 = tf.divide(sim, norms)
    div2 = tf.divide(div1, normst)
    result = sess.run(div2)
    sess.close()
    return result


def comp_similarity_tf2(ratings):
    print("  similaridade tensorflow com placeholder")
    sess = tf.Session()
    
    matrix1 = tf.placeholder(tf.float32, shape=ratings.shape)
    #matrix1 = tf.convert_to_tensor(ratings, np.float32)
    #matrix2 = tf.transpose(matrix1)
    sim = tf.matmul(matrix1, matrix1, transpose_b=True, a_is_sparse=True, b_is_sparse=True)
    diag0 = tf.diag_part(sim)
    diag = tf.zeros([1, diag0.get_shape()[0]], tf.float32)
    diag = tf.add(diag, diag0)
    norms = tf.sqrt(diag)
    normst = tf.transpose(norms)
    div1 = tf.divide(sim, norms)
    div2 = tf.divide(div1, normst)
    result = sess.run(div2, feed_dict={matrix1: ratings.toarray()})
    sess.close()
    return result

def get_similar_from_busca(dados, idx, N=5, minimoinicial=0.3, minimofinal=0.2):
    #exrr = np.array([  848,   1830,  1950,  1913,  2428,  3542, 3558, 3561, 3576, 3598, 3622,
    #                   3626, 3627, 3636, 3641, 3683, 3718,  3720,  3723, 3746, 3730, 
    #                   3770, 4391,  4560,  5201,  5264,  5478,  5664,  5881,  5918,  
    #                   6119,  6190,  6539,  6983,  7286])

    if idx>=len(dados["busca"]["itens"]):
        print("  Índice fornecido é de material:", idx)
        return [], [], []
    row = dados["sim"][idx, :]
    imgs = []
    mids = []
    indx = []
    valsim = []
    cont=0
    
    #mediadoc = np.average(row[exrr])
    #maxdoc = np.max(row[exrr])
    
    #if maxdoc>0.4:
    #    minimoinicial=0.85
    #    minimofinal=0.7

    if idx==8555:
        print("  idx=", idx)
        print("  matches ", dados["total"]["nomes"][idx])
        print("  ", np.argsort(-row)[0:30])
    for x in np.argsort(-row):
        if x>=len(dados["busca"]["itens"]):
            #if idx == 8555:
            #    print("x=", x, "row[x]=", row[x])
            if ((cont==0 and row[x]<minimoinicial) or row[x]<minimofinal): break
            #print(row[x])
            imgs.append(dados["total"]["nomes"][x])
            mids.append(dados["total"]["itens"][x][0])
            valsim.append(row[x])
            indx.append(x)
            cont += 1
            if cont>=N: break
    #if len(imgs)>0: print(idx, " - correlação documento:", mediadoc, maxdoc)
    return imgs, mids, valsim, indx


#relatório procura no material imagens similares às da busca
def gera_html_busca(dados, arqs, excl, N=5, minimoinicial=0.3, minimofinal=0.2):
    cont=0
    contsim=0
    nro_arq_busca = 0
    try:
        os.stat("relatorios")
    except:
        os.mkdir("relatorios")    

    try:
        os.stat("relatorios/imgreport_b")
    except:
        os.mkdir("relatorios/imgreport_b")    
        
    outF = open("relatorios/similares_a_busca.html", "w", encoding='utf-8')
    outF.write("<html>\n")
    outF.write("<head><meta charset=\"UTF-8\"></head>\n")
    outF.write("<style> \n\
    table {\n\
        border-collapse: collapse;\n\
        width: 100%;\n\
        background-color: #ffffff;\n\
    }\n\
    th, td {\n\
        border: 1px solid #ccc;\n\
        padding: 10px;\n\
    }\n\
    th {\n\
      background-color: #dadbda;\n\
    }\n\
    table.alt tr:nth-child(even) {\n\
        background-color: #eee;\n\
    }\n\
    table.alt tr:nth-child(odd) {\n\
        background-color: #fff;\n\
    }\n\
    .tabelamaior {\n\
       background-color: #fafbfa;\n\
     }\n\
    .titulo1 {\n\
	background-color: #dccccc;\n\
	height:60px;\n\
    }\n\
    .titulo2 {\n\
	background-color: #ccdccc;\n\
	height:60px;\n\
    }\n\
</style>\n")    
    
    outF.write("<body>\n")

    for ind in range (0, len(dados["busca"]["itens"])):
        if ind in excl: continue
        resimg, resmid, resvalsim, indx = get_similar_from_busca(dados, ind, N=N, minimoinicial=minimoinicial, minimofinal=minimofinal)
        if "20150519_095035.jpg" in dados["total"]["nomes"][ind]:
            print("  ", dados["total"]["nomes"][ind])
            print("  ", ind, resimg, resmid, resvalsim, indx)
        nro_arq_busca += 1
        if len(resimg)>0:
            #print(resimg, resmid, resvalsim, indx)
            try:
                img = mpimg.imread(dados["total"]["nomes"][ind])
                max = min(img.shape[0], img.shape[1])
                if max<40: 
                    #print("max<40", img.shape, dados["total"]["nomes"][ind])
                    continue
                outF.write("<table class=tabelamaior style=\"padding:10px\" valign=top><tr><th width=50%>Imagem do " + 
			arqs["busca"]["nome_exibicao"] + "</th><th width=50%>Imagens similares no " + 
			arqs["material"]["nome_exibicao"] + "</th></tr>\n")
                outF.write("<tr>\n\
      <td valign=top>\n\
      <table valign=top>\n")
                outF.write("    <tr><th class=titulo1 >" + dados["total"]["nomes"][ind][len(arqs["busca"]["raiz"]):] + "</th></tr>\n")
                outF.write("    <tr><td>Índice: " + str(ind) + "</td></tr>\n")
                outF.write("    <tr><td>Tamanho (bytes): " 
                           + str(os.path.getsize(dados["total"]["nomes"][ind])) + "</td></tr>\n")
                outF.write("    <tr><td>Dimensões da imagem (alturaxlargura): " + 
                             str(img.shape[0]) + "x" + str(img.shape[1]) + "</td></tr>\n")
                outF.write("    <tr><td>Hash (MD5): <font style=\"font-family:courier;\">" + 
                           dados["total"]["itens"][ind][0] + "</font></td></tr>\n")



                h = img.shape[0]
                w = img.shape[1]
                szmax=400
                if h>w and h>szmax: 
                    w = int(w * szmax / h)
                    h=szmax
                if w>h and w>szmax: 
                    h = int(h * szmax / w)
                    w=szmax

                novonome = "imgreport_b/" + dados["total"]["itens"][ind][0] + dados["total"]["nomes"][ind][dados["total"]["nomes"][ind].rfind("."):]
                try:
                    copyfile(dados["total"]["nomes"][ind], "relatorios/"+novonome)
                except:
                    print("  erro no copyfile de", novonome)

                #outF.write("    <tr><td > <img src='" + dados["total"]["nomes"][ind].replace("%", "%25") + 
                outF.write("    <tr><td > <img src='" + novonome + 
                           "' width=" + str(w) + "px heigth=" + str(h) + "px></td></tr>\n")
                outF.write("  </table>\n  </td>\n  <td valign=top>\n")

                cont+=1

                for res in range(0, len(resimg)):
                    nomeimg = resimg[res]
                    #print(nomeimg)
                    valsim = resvalsim[res]
                    #if res==0 and valsim<nivel: limiar.append(ind)
                    try:
                        img = mpimg.imread(nomeimg)
                        contsim+=1

                        outF.write("  <table valign=top>\n")

                        outF.write("    <tr><th class=titulo2 >" + str(res+1) + "a imagem similar: " +
                                   nomeimg[len(arqs["material"]["raiz"]):] + "</th></tr>\n")

                        outF.write("    <tr><td>Índice: " + str(indx[res]) + " (Similaridade: " +
                                   "{:2.3f}".format(valsim) + ")</td></tr>\n")

                        outF.write("    <tr><td>Tamanho (bytes): " 
                                   + str(os.path.getsize(nomeimg)) + "</td></tr>\n")
                        outF.write("    <tr><td>Dimensões da imagem (alturaxlargura): " + 
                                     str(img.shape[0]) + "x" + str(img.shape[1]) + "</td></tr>\n")
                        outF.write("    <tr><td>Hash (MD5): <font style=\"font-family:courier;\">" + 
                                   dados["total"]["itens"][indx[res]][0] + "</font></td></tr>\n")


                        h = img.shape[0]
                        w = img.shape[1]
                        szmax=400
                        if h>w and h>szmax: 
                            w = int(w * szmax / h)
                            h=szmax
                        if w>h and w>szmax: 
                            h = int(h * szmax / w)
                            w=szmax
                        novonome = "imgreport_b/" + dados["total"]["itens"][indx[res]][0] + nomeimg[nomeimg.rfind("."):]
                        try:
                            copyfile(nomeimg, "relatorios/"+novonome)
                        except:
                            print("  erro no copyfile de", novonome)

                        #outF.write("    <tr><td > <img src='" + nomeimg.replace("%", "%25") + 
                        outF.write("    <tr><td > <img src='" + novonome + 
                                   "' width=" + str(w) + "px heigth=" + str(h) + "px></td></tr>\n")
                        outF.write("  </table>\n  <br>\n")
                    except Exception as ex2:
                        print("\n  Erro ", ex2)
                        print("   -- imreag ", nomeimg)


                outF.write("  </td>\n</tr>\n</table>\n<br>\n\n")                

                outF.write("<hr style=\"border=2\">")

                #if cont>30: break
            
            except Exception as eximread:
                print("  Erro ", eximread) 
                print("   -- imreag ", dados["total"]["nomes"][ind])
    outF.write("</body>\n")
    outF.write("</html>\n")
    outF.close()
    print("  ", cont, contsim, nro_arq_busca)    
    


def get_similar_from_material(dados, idx, N=5, minimoinicial=0.3, minimofinal=0.2):
    #exrr = np.array([  848,   1830,  1950,  1913,  2428,  3542, 3558, 3561, 3576, 3598, 3622,
    #                   3626, 3627, 3636, 3641, 3683, 3718,  3720,  3723, 3746, 3730, 
    #                   3770, 4391,  4560,  5201,  5264,  5478,  5664,  5881,  5918,  
    #                   6119,  6190,  6539,  6983,  7286])

    if idx<len(dados["busca"]["itens"]):
        print("  Índice fornecido é de busca:", idx)
        return [], [], []
    row = dados["sim"][idx, :]
    imgs = []
    mids = []
    indx = []
    valsim = []
    cont=0
    
    #mediadoc = np.average(row[exrr])
    #maxdoc = np.max(row[exrr])
    
    #if maxdoc>0.4:
    #    minimoinicial=0.85
    #    minimofinal=0.7
    if idx==8555:
        print("  idx=", idx)
        print("  matches ", dados["total"]["nomes"][idx])
        print("  ", np.argsort(-row)[0:30])
    for x in np.argsort(-row):
        if x<len(dados["busca"]["itens"]):
            #if idx == 8555:
            #    print("x=", x, "row[x]=", row[x])
            if ((cont==0 and row[x]<minimoinicial) or row[x]<minimofinal): break
            #print(row[x])
            imgs.append(dados["total"]["nomes"][x])
            mids.append(dados["total"]["nomes"][x][0])
            valsim.append(row[x])
            indx.append(x)
            cont += 1
            if cont>=N: break
    #if len(imgs)>0: print(idx, " - correlação documento:", mediadoc, maxdoc)
    return imgs, mids, valsim, indx

#relatório procura na busca imagens similares às do material 
def gera_html_material(dados, arqs, excl, N=5, minimoinicial=0.3, minimofinal=0.2):
    cont=0
    contsim=0

    try:
        os.stat("relatorios")
    except:
        os.mkdir("relatorios")    

    try:
        os.stat("relatorios/imgreport_m")
    except:
        os.mkdir("relatorios/imgreport_m")    
        
    outF = open("relatorios/similares_ao_material.html", "w", encoding='utf-8')
    outF.write("<html>\n")
    outF.write("<head><meta charset=\"UTF-8\"></head>\n")
    outF.write("<style> \n\
    table {\n\
        border-collapse: collapse;\n\
        width: 100%;\n\
        background-color: #ffffff;\n\
    }\n\
    th, td {\n\
        border: 1px solid #ccc;\n\
        padding: 10px;\n\
    }\n\
    th {\n\
      background-color: #dadbda;\n\
    }\n\
    table.alt tr:nth-child(even) {\n\
        background-color: #eee;\n\
    }\n\
    table.alt tr:nth-child(odd) {\n\
        background-color: #fff;\n\
    }\n\
    .tabelamaior {\n\
       background-color: #fafbfa;\n\
     }\n\
    .titulo1 {\n\
	background-color: #dccccc;\n\
	height:60px;\n\
    }\n\
    .titulo2 {\n\
	background-color: #ccdccc;\n\
	height:60px;\n\
    }\n\
</style>\n")    
    
    outF.write("<body>\n")

    for ind in range (len(dados["busca"]["itens"]), len(dados["total"]["itens"])):
        if ind in excl: continue
        resimg, resmid, resvalsim, indx = get_similar_from_material(dados, ind, N=N, minimoinicial=minimoinicial, minimofinal=minimofinal)
        #if "20150519_095035.jpg" in dados["total"]["nomes"][ind]:
        #    print(dados["total"]["nomes"][ind])
        #    print(ind, resimg, resmid, resvalsim, indx)
        if len(resimg)>0:
            #print("  ", dados["total"]["nomes"][ind])
            try:
                img = mpimg.imread(dados["total"]["nomes"][ind])
                max = min(img.shape[0], img.shape[1])
                if max<40: 
                    #print("max<40", img.shape, dados["total"]["nomes"][ind])
                    continue
                outF.write("<table class=tabelamaior style=\"padding:10px\" valign=top><tr><th width=50%>Imagem do " + 
			arqs["material"]["nome_exibicao"] + "</th><th width=50%>Imagens similares no " +
			arqs["busca"]["nome_exibicao"] + "</th></tr>\n")
                outF.write("<tr>\n\
      <td valign=top>\n\
      <table valign=top>\n")
                outF.write("    <tr><th class=titulo1 >" + dados["total"]["nomes"][ind][len(arqs["material"]["raiz"]):] + "</th></tr>\n")
                outF.write("    <tr><td>Índice: " + str(ind) + "</td></tr>\n")
                outF.write("    <tr><td>Tamanho (bytes): " 
                           + str(os.path.getsize(dados["total"]["nomes"][ind])) + "</td></tr>\n")
                outF.write("    <tr><td>Dimensões da imagem (alturaxlargura): " + 
                             str(img.shape[0]) + "x" + str(img.shape[1]) + "</td></tr>\n")
                outF.write("    <tr><td>Hash (MD5): <font style=\"font-family:courier;\">" + 
                           dados["total"]["itens"][ind][0] + "</font></td></tr>\n")



                h = img.shape[0]
                w = img.shape[1]
                szmax=400
                if h>w and h>szmax: 
                    w = int(w * szmax / h)
                    h=szmax
                if w>h and w>szmax: 
                    h = int(h * szmax / w)
                    w=szmax

                novonome = "imgreport_m/" + dados["total"]["itens"][ind][0] + dados["total"]["nomes"][ind][dados["total"]["nomes"][ind].rfind("."):]
                try:
                    copyfile(dados["total"]["nomes"][ind], "relatorios/"+novonome)
                except:
                    print("  erro no copyfile de", novonome)

                #outF.write("    <tr><td > <img src='" + dados["total"]["nomes"][ind].replace("%", "%25") + 
                outF.write("    <tr><td > <img src='" + novonome + 
                           "' width=" + str(w) + "px heigth=" + str(h) + "px></td></tr>\n")
                outF.write("  </table>\n  </td>\n  <td valign=top>\n")

                cont+=1



                for res in range(0, len(resimg)):
                    nomeimg = resimg[res]
                    #print(nomeimg)
                    valsim = resvalsim[res]
                    #if res==0 and valsim<nivel: limiar.append(ind)
                    try:
                        img = mpimg.imread(nomeimg)
                        contsim+=1

                        outF.write("  <table valign=top>\n")

                        outF.write("    <tr><th class=titulo2 >" + str(res+1) + "a imagem similar: " +
                                nomeimg[len(arqs["busca"]["raiz"]):] + "</th></tr>\n")

                        outF.write("    <tr><td>Índice: " + str(indx[res]) + " (Similaridade: " +
                                "{:2.3f}".format(valsim) + ")</td></tr>\n")

                        outF.write("    <tr><td>Tamanho (bytes): " 
                                + str(os.path.getsize(nomeimg)) + "</td></tr>\n")
                        outF.write("    <tr><td>Dimensões da imagem (alturaxlargura): " + 
                                    str(img.shape[0]) + "x" + str(img.shape[1]) + "</td></tr>\n")
                        outF.write("    <tr><td>Hash (MD5): <font style=\"font-family:courier;\">" + 
                                dados["total"]["itens"][indx[res]][0] + "</font></td></tr>\n")


                        h = img.shape[0]
                        w = img.shape[1]
                        szmax=400
                        if h>w and h>szmax: 
                            w = int(w * szmax / h)
                            h=szmax
                        if w>h and w>szmax: 
                            h = int(h * szmax / w)
                            w=szmax
                        novonome = "imgreport_m/" + dados["total"]["itens"][indx[res]][0] + nomeimg[nomeimg.rfind("."):]
                        try:
                            copyfile(nomeimg, "relatorios/"+novonome)
                        except:
                            print("  erro no copyfile de", novonome)

                        #outF.write("    <tr><td > <img src='" + nomeimg.replace("%", "%25") + 
                        outF.write("    <tr><td > <img src='" + novonome + 
                                "' width=" + str(w) + "px heigth=" + str(h) + "px></td></tr>\n")
                        outF.write("  </table>\n  <br>\n")
                    except Exception as ex2:
                        print("\n  Erro ", ex2)
                        print("   -- imreag ", nomeimg)


                outF.write("  </td>\n</tr>\n</table>\n<br>\n\n")                

                outF.write("<hr style=\"border=2\">")

                #if cont>30: break
            
            except Exception as eximread:
                print("  Erro ", eximread) 
                print("   -- imreag ", dados["total"]["nomes"][ind])
    outF.write("</body>\n")
    outF.write("</html>\n")
    outF.close()
    print("  ", cont, contsim)    


