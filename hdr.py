from PySide2.QtGui import QAccessibleEvent
import PySimpleGUI as gui
import os, sys
from os import path
from PySimpleGUIQt.PySimpleGUIQt import Stretch, theme_background_color
import cv2
import numpy as np
from numpy.core.fromnumeric import size

from numpy.lib.function_base import kaiser
from numpy.lib.shape_base import expand_dims

gui.theme_element_background_color('#2d2d2d')
gui.theme_background_color('#2d2d2d')
gui.theme_element_text_color('#666666')

#--------------------------------------------#
def resize(file, largura, altura):
    cwd_path=pathF()
    img=cv2.imread(file)

    width=img.shape[1]
    height=img.shape[0]
    
    
    if width>height:
        reduction=largura/width #Peso do redimensionamento de modo a não distorcer a imagem
        if reduction>1: #Aumentar tamanho da imagem
            dim=(largura, int(height*reduction))
            img_red=cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC) #Redimensionamento das imagens com interpolação do tipo CUBIC (melhor para aumento de imagens)
        else: #Diminuir tamanho da imagem
            dim=(largura, int(height*reduction))
            img_red=cv2.resize(img, dim, interpolation=cv2.INTER_AREA) #Redimensionamento das imagens com interpolação do tipo AREA (melhor para redução de imagens)
            
        temp_path=os.path.join(cwd_path,"Temp/", "temp_img.png") #Guarda a imagem num caminho temporário, apenas para a mostrar no ecrã
        cv2.imwrite(temp_path, img_red)
        return temp_path
    
    elif width<height:
        reduction=altura/height
        if reduction>1:
            dim=(int(width*reduction), altura)
            img_red=cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        else:
            dim=(int(width*reduction), altura)
            img_red=cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        temp_path=os.path.join(cwd_path,"Temp/", "temp_img.png")
        cv2.imwrite(temp_path, img_red)
        return temp_path
    
    else: #Imagens quadradas
        reduction=altura/height
        if reduction>1:
            dim=(int(width*reduction), int(height*reduction))
            img_red=cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        else:
            dim=(int(width*reduction), int(height*reduction))
            img_red=cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        temp_path=os.path.join(cwd_path,"Temp/", "temp_img.png")
        cv2.imwrite(temp_path, img_red)
        return temp_path

def debevechdr(path2):
    cwd_path=pathF()

    #Leitura das imagens e dos tempos de exposição na formatação float32
    times = np.array([ 1/1000.0, 1/125.0, 1/8.0, 1.0 ], dtype=np.float32)
    filenames=os.listdir(path2) #Lista com o nome dos ficheiros no caminho anterior
    
    images=[] #Lista onde vão ser colocadas as imagens encontradas na pasta
    for i in range(len(filenames)):
        img=os.path.join(path2, filenames[i]) #Caminho da imagem i
        im=cv2.imread(img) #Leitura da imagem i
        images.append(im) #Inserção do array de imagem na lista
    
    align=cv2.createAlignMTB()
    align.process(images, images) #Alinhamento das imagens

    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times)  #Aplicação do algoritmo de Debevec

    #Tonemapping (passamos as imagens HDR de 32bit float para 8bit inteiro com range [0, 255])
    tonemapping = cv2.createTonemap(gamma=3.0)
    res_debevec = tonemapping.process(hdrDebevec.copy())

    res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8') #Conversão novamente para 8bit para poder ser visto em ecrãs normais

    proc_path=os.path.join(cwd_path, "Results/", "imgProcessed.png") #Caminho onde guardar
    cv2.imwrite(proc_path, res_debevec_8bit) #Ação de guardar

    return proc_path

def mertenshdr(path2):
    cwd_path=pathF()

    filenames=os.listdir(path2) #Lista com o nome dos ficheiros no caminho anterior
    
    images=[] #Lista onde vão ser colocadas as imagens encontradas na pasta
    for i in range(len(filenames)): 
        img=os.path.join(path2, filenames[i]) #Caminho da imagem i
        im=cv2.imread(img) #Leitura da imagem i
        images.append(im) #Inserção do array de imagem na lista

    align=cv2.createAlignMTB()
    align.process(images, images) #Alinhamento das imagens

    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(images) #Aplicação do algoritmo de Mertens

    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8') #Para podermos mostrar as imagens, temos de as converter para 8bit de 0 a 255 

    proc2_path=os.path.join(cwd_path, "Results/", "imgProcessed2.png")
    cv2.imwrite(proc2_path, res_mertens_8bit)

    return proc2_path

def gammaPro(im, gamma, save, j): #Função responsável por fazer a correção de gamma (o funcionamento é igual à função BnC())
    cwd_path=pathF()

    img=cv2.imread(im)
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)

    if save==1:
        if j==0:
            proc3_path=os.path.join(cwd_path, "Results/myHDR/", "myHdrPr1.png")
            cv2.imwrite(proc3_path, res)
            return proc3_path 
        elif j==1:
            proc3_path=os.path.join(cwd_path, "Results/myHDR/", "myHdrPr2.png")
            cv2.imwrite(proc3_path, res)
            return proc3_path 
        elif j==2:
            proc3_path=os.path.join(cwd_path, "Results/myHDR/", "myHdrPr3.png")
            cv2.imwrite(proc3_path, res)
            return proc3_path
    elif save==0:
        proc3_path=os.path.join(cwd_path, "Temp/", "TempImgProcessed.png")
        cv2.imwrite(proc3_path, res)
        return proc3_path
    
def BnC(im, alpha, beta, save, j): #Função responsável por alterar o contraste e o brightness
    cwd_path=pathF()
    image = cv2.imread(im)
    new_image = np.zeros(image.shape, image.dtype)
    
    #Acedemos a cada píxel da imagem e aplicamos a operação g(x)=alpha*f(x)+beta, onde alpha é o ganho (contraste) e beta o bias (brightness)
    for y in range(image.shape[0]): #Linha do array imagem
        for x in range(image.shape[1]): #Coluna do array imagem
            for c in range(image.shape[2]): #Componente R ou G ou B
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)


    if save==1: #save==1 significa guardar a imagem processada
        if j==0: #O j serve para manter a função a par de quantas imagens já foram guardadas, visto que só queremos 3
            proc4_path=os.path.join(cwd_path, "Results/myHDR/", "myHdrPr1.png")
            cv2.imwrite(proc4_path, new_image)
            return proc4_path 
        elif j==1:
            proc4_path=os.path.join(cwd_path, "Results/myHDR/", "myHdrPr2.png")
            cv2.imwrite(proc4_path, new_image)
            return proc4_path 
        elif j==2:
            proc4_path=os.path.join(cwd_path, "Results/myHDR/", "myHdrPr3.png")
            cv2.imwrite(proc4_path, new_image)
            return proc4_path
    elif save==0: #save==0 significa não guardar a imagem na pasta definitiva, ao invés, guarda temporariamente numa pasta que serve apenas para mostrar a imagem 
        proc4_path=os.path.join(cwd_path, "Temp/", "TempImgProcessed.png")
        cv2.imwrite(proc4_path, new_image)
        return proc4_path

def merge(): #Função que faz a média pesada das 3 imagens produzidas pelo utilizador
    cwd_path=pathF()
    path2=os.path.join(cwd_path, "Results/myHDR") #Caminho onde estão os ficheiros
    filenames=os.listdir(path2) #Lista com o nome dos ficheiros no caminho anterior
    images=[] #Lista onde vão ser colocadas as imagens encontradas na pasta
    for i in range(len(filenames)):
        img=os.path.join(path2, filenames[i]) #Caminho da imagem i
        im=cv2.imread(img) #Leitura da imagem i
        images.append(im) #Inserção do array de imagem na lista

    res = images[0]
    for i in range(len(images)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1) #Peso da imagem i
            beta = 1.0 - alpha #Peso do resultado
            res = cv2.addWeighted(images[i], alpha, res, beta, 0.0) #Junta as imagens (média pesada)
    
    proc4_path=os.path.join(cwd_path, "Results/", "mergeMyHDR.png") #Caminho onde guardar
    cv2.imwrite(proc4_path, res) #Ação de guardar a imagem no caminho especificado
    return proc4_path

def pathF(): #Função que dá o "current working directory" automaticamente, não sendo preciso especificar onde está nada... 
    if getattr(sys, 'frozen', False): #...consoante o código é corrido no .exe...
        cwd_path = os.path.dirname(sys.executable)
        print(cwd_path)
        return cwd_path
    else: #... ou diretamente no script.
        cwd_path = sys.path[0]
        return cwd_path
#--------------------------------------------#
image_search = [
    [
    gui.Text('Escolha uma pasta...', font=('Helvetica', 10, 'bold'), background_color='#2d2d2d'),
    gui.In(size=(25, 1), enable_events=True, key='Search'), #Input
    gui.FolderBrowse(button_text='Procurar', size=(8, 1), button_color=('#ffffff', '#f4d47c'), font=('Helvetica', 10, 'bold'), tooltip='Navegue entre as pastas e selecione a respetiva com as imagens a serem processadas! \n'
                                                                                                                                        '\n-> Selecione Upload para correr o programa como é suposto'
                                                                                                                                        '\nNota: É possível selecionar qualquer pasta, porém, podem não ser reunidas as condições' 
                                                                                                                                        '\npara os 2 primeiros algoritmos de processamento funcionarem. A manipulação da imagem' 
                                                                                                                                        '\nna terceira aba e o resultado da quarta aba, respetivamente, funcionam sempre!')
                                                                                                                                        
    ],
    [
    gui.Listbox(
        values=[], size=(55, 10), enable_events=True, key='List', pad=(5,5))
    ],
    [
        gui.Button('Image Slide Viewer', size=(18, 1), button_color=('#ffffff', '#f4d47c'), font=('Helvetica', 10, 'bold'), key='ImgPrevbutton', pad=((5,0),(0,25)), tooltip=' Visualizador das images da lista em slide ')
    ],
    [gui.Frame('',[[gui.Text('Image Previewer', font=('Helvetica', 15, 'bold'), background_color='#2d2d2d', tooltip=' Pré-visualização da imagem selecionada na lista superior ')],
    [gui.Text(size=(50, 1), key='TextOut', background_color='#2d2d2d')],
    [gui.Image(size=(500, 375), key='Image')]], pad=(0,0))],
]

proc_disp1 = [
    [gui.Text('Debevec Algorithm', font=('Helvetica', 15, 'bold'), background_color='#2d2d2d', justification='center')],
    [gui.Text('', background_color='#2d2d2d')],
    [gui.Image(size=(850, 650), key='ImageP')]
]

proc_disp2 = [
    [gui.Text('Mertens Fusion Algorithm', font=('Helvetica', 15, 'bold'), background_color='#2d2d2d', justification='center')],
    [gui.Text('', background_color='#2d2d2d')],
    [gui.Image(size=(850, 650), key='ImageP2')]
]

proc_disp3 = [
    [gui.Text('Single Image HDR - Image Processing', font=('Helvetica', 15, 'bold'), background_color='#2d2d2d', justification='center')],
    [gui.Text('', background_color='#2d2d2d')],
    [gui.Image(size=(850, 650), key='ImageP3')]
]

proc_disp4 = [
    [gui.Text('Single Image HDR - Result', font=('Helvetica', 15, 'bold'), background_color='#2d2d2d', justification='center')],
    [gui.Text('', background_color='#2d2d2d')],
    [gui.Image(size=(850, 650), key='ImageP4')]
]

tab1_layout=[[gui.Frame('', proc_disp1, border_width=0, element_justification='center', size=(1000, 700), pad=(0, 0))]]
tab2_layout=[[gui.Frame('', proc_disp2, border_width=0, element_justification='center', size=(1000, 700), pad=(0, 0))]]
tab3_layout=[[gui.Frame('', proc_disp3, border_width=0, element_justification='center', vertical_alignment='center', size=(1000, 700), pad=(0, 0))]]
tab4_layout=[[gui.Frame('', proc_disp4, border_width=0, element_justification='center', vertical_alignment='center', size=(1000, 700), pad=(0, 0))]]


layout=[
    [gui.Frame('', image_search, element_justification='left', size=(580, 700), border_width=0, vertical_alignment='top', pad=((0,5),0)), 
    gui.VSeperator(), 
    gui.TabGroup([[
    gui.Tab('Processing #1', tab1_layout, background_color='#2d2d2d', font=('Helvetica', 10, 'bold'), tooltip='Aqui é mostrado o resultado do processamento por Debevec'), 
    gui.Tab('Processing #2', tab2_layout, background_color='#2d2d2d', font=('Helvetica', 15, 'bold'), tooltip='Aqui é mostrado o resultado do processamento por Mertens'), 
    gui.Tab('Processing #3', tab3_layout, background_color='#2d2d2d', font=('Helvetica', 15, 'bold'), element_justification='center', tooltip='Aqui é feita a manipulação da imagem a ser processada em #4'),
    gui.Tab('Processing #4', tab4_layout, background_color='#2d2d2d', font=('Helvetica', 15, 'bold'), element_justification='center', tooltip='Aqui é mostrado o resultado do processamento Single Image HDR')]], 
    tab_background_color='#f4d47c', selected_background_color='#16394f', title_color='#ffffff', selected_title_color='#ffffff', font=('Helvetica', 10, 'bold'))],
]

window = gui.Window("HDR Processing (COM TOOLTIPS)", layout, resizable=False).Finalize()

image_viewer_slide=[
    [gui.Text('Image Previewer', font=('Helvetica', 10, 'bold'), background_color='#2d2d2d')],
    [gui.Text(size=(100,1), justification='center', key='-TextOut2-', background_color='#2d2d2d')],
    [gui.Image(key='-ImagePrev-')],
]

layout2=[
    [gui.Button('<-', size=(5, 1), button_color=('#ffffff', '#f4d47c'), key='-Left-'), gui.Column(image_viewer_slide, element_justification='c'), gui.Button('->', size=(5, 1), button_color=('#ffffff', '#f4d47c'), key='-Right-')],
]

window2 = gui.Window("Image Slide Viewer", layout2, element_justification='center', return_keyboard_events=True)

slider_layout=[[gui.T('Por favor, abra a aba Processing #3!\n\nObjetivo: Altere e guarde 3 imagens de modo a que: \nUma tenha pouca exposição; \nUma esteja exposta corretamente; \nUma esteja demasiado exposta.\n O resultado da média pesada destas 3 será mostrado na aba #4', 
                        background_color='#2d2d2d', font=('Helvetica', 10, 'bold'), justification='center')],
        [gui.T('O valor de Gamma é...')],
        [gui.Column([[gui.Slider(range=(0.1, 10), orientation='h', size=(30,20), default_value=1.0, resolution=0.1, enable_events=True, disable_number_display=False, key='Slider'), gui.Button('Aplicar', pad=(0, (20, 0)), enable_events=True, key='Button1')]])],
        [gui.T('O valor de Contraste é...')],
        [gui.Column([[gui.Slider(range=(1, 3), orientation='h', size=(30,20), default_value=1.0, resolution=0.1, enable_events=True, disable_number_display=False, key='Slider2'), gui.Button('Aplicar', pad=(0, (20, 0)), enable_events=True, key='Button2')]])],
        [gui.T('O valor de Brightness é...')],
        [gui.Column([[gui.Slider(range=(0, 100), orientation='h', size=(30,20), default_value=0, resolution=1, enable_events=True, disable_number_display=False, key='Slider3'), gui.Button('Aplicar', pad=(0, (20, 0)), enable_events=True, key='Button3')]])],
        [gui.Button('Guardar imagem', auto_size_button=True, enable_events=True, key='ButtonApp'), gui.Button('Sair', auto_size_button=True, enable_events=True, key='ButtonExit')]
]

window3=gui.Window('Painel de Controlo', slider_layout)

aux=[]
while True:
    event, values = window.read() #Lê tudo o que se passa na janela
    if event=='Exit' or event==gui.WIN_CLOSED: #Casos em que o utilizador fecha a janela
        break

    if event=='Search': #Detetada a key de procura de uma pasta
        search=values['Search'] #É criada uma lista com os valores do Input (neste caso, o caminho da pasta)
        try:
            search_list_files=os.listdir(search) #Obtém a lista de ficheiros dentro da pasta escolhida
        except:
            search_list_files=[] #Lista vazia no caso de não se verificar a key 'Search'

        name_list_files = [ #Lista onde vão ser colocados os nomes dos ficheiros e o caminho
            f
            for f in search_list_files
            if os.path.isfile(os.path.join(search, f)) #Verifica se o argumento é um ficheiro
                 #Junta os caminhos, neste caso, search+f será algo do tipo: (Caminho)+(nome do ficheiro)
        ]
        
        if 'Upload' in search: #Apenas será feito o processamento caso as imagens estejam na pasta upload
            window['ImageP'].update(resize(debevechdr(search), 850, 600)) #Apresentação do resultado de Debevec na janela (separador 1)
            window['ImageP2'].update(resize(mertenshdr(search), 850, 600)) #Apresentação do resultado de Mertens na janela (separador 2)
        else:
            pass


        window['List'].update(name_list_files) #Display na janela da lista de ficheiros da pasta selecionada
     
        

    elif event=='List': #Detetada a key de seleção de um ficheiro na lista mostrada
        im_path=os.path.join(values['Search'], values['List'][0]) #Caminho onde está a imagem selecionada na lista
        img2=cv2.imread(im_path) #Leitura da imagem
        if ((img2.shape[0]>500 and img2.shape[1]>1000)):
            try:
                window['TextOut'].update(im_path)        
                window['Image'].update(resize(im_path, 500, 700)) #Mostra a imagem redimensionada graças à função resize()
            except:
                pass
        elif ((img2.shape[0]==img2.shape[1]) and img2.shape[0]>500): #Caso de imagens quadradas
            window['TextOut'].update(im_path)        
            window['Image'].update(resize(im_path, 500, 500))
        else: #Todos os outros casos
            try:
                window['TextOut'].update(im_path)
                window['Image'].update(im_path)
            except:
                pass
        
        i=len(aux)
        save=0
        window3.Finalize()
        window['ImageP3'].update(resize(im_path, 850, 600))
        a=True
        while a:
            event3, values3 = window3.read()
            
            try:
                gamma=values3.get('Slider') #Gamma correction
                alpha=values3.get('Slider2') #Contraste
                beta=values3.get('Slider3') #Brightness
            except: #Caso não seja feita uma seleção, inicializa com os valores "normais"
                gamma=1.0 
                alpha=1.0
                beta=0.0
            
            if event3==None or event3==gui.WIN_CLOSED: #Casos em que o utilizador fecha a janela
                break

            if event3=='Button1': #Apenas é feita a alteração da gamma (graças à função gammaPro()), se o utilizador selecionar "Aplicar" -> Button1 (primeira linha)
                save=0 #Não guarda logo a imagem, apenas se o utilizador selecionar o botão "Aplicar e guardar" -> ButtonApp
                img2resize=gammaPro(im_path, gamma, save, i)
                window['ImageP3'].update(resize(img2resize, 850, 600))
            
            elif event3=='Button2':
                save=0
                img2resize=BnC(im_path, alpha, beta, save, i) #Apenas é feita a alteração do contraste (graças à função BnC()), se o utilizador selecionar "Aplicar" -> Button2 (segunda linha)
                window['ImageP3'].update(resize(img2resize, 850, 600))
            
            elif event3=='Button3':
                save=0 
                img2resize=BnC(im_path, alpha, beta, save, i) #Apenas é feita a alteração da brightness (graças à função BnC()), se o utilizador selecionar "Aplicar" -> Button3 (terceira linha)
                window['ImageP3'].update(resize(img2resize, 850, 600))

            elif event3=='ButtonApp':
                save=1 #Guarda a imagem
                print(gamma)
                aux.append(i)
                pass2fun=gammaPro(im_path, gamma, save, i) #Chama novamente a função gammaPro com os novos parâmetros
                i=i+1
                if i==3: #Havendo 3 imagens guardadas...
                    window3.close()
                    window['ImageP4'].update(resize(merge(), 850, 600)) #...é mostrada no separador #4 o resultado da média pesada destas (graças à função merge())
                    break

            elif event3=='ButtonExit':
                save=0
                window3.close()
                break

            else:
                continue
            
    
    elif event=='ImgPrevbutton': #Detetada a key de seleção do visualizador em slide
        if values['Search']!='':
            window2.finalize()
            i=0
            while True:
                nf = [ #Lista onde vão ser colocados os nomes dos ficheiros e o caminho
                    f2
                    for f2 in os.listdir(values['Search'])
                    if os.path.isfile(os.path.join(values['Search'], f2)) #Verifica se o argumento é um ficheiro
                        #Junta os caminhos, neste caso, search+f será algo do tipo: (Caminho)+(nome do ficheiro)
                ]
                fileP=os.path.join(values['Search'], nf[i])
                window2['-TextOut2-'].update(fileP)
                window2['-ImagePrev-'].update(resize(fileP, 800, 800))
                event2 = window2.read()
                
                
                if list(event2)[0]==None or list(event2)[0]==gui.WIN_CLOSED: #Casos em que o utilizador fecha a janela
                    event=window.read()
                    break
                
                elif list(event2)[0]=='-Right-' and i < len(nf)-1: #Mostra a imagem seguinte
                    i = i + 1

                elif list(event2)[0]=='-Left-' and i > 0: #Mostra a imagem anterior
                    i = i - 1  
        else: 
            print('Erro: Procurar pasta com as imagens desejadas')
            
window.close()        