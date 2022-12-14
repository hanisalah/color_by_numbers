import cv2
import extcolors
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
from copy import deepcopy
from zipfile import ZipFile
from datetime import datetime

def ix_change(mode=0):
    """manipulate index of image to be shown based on clicks from relevant prev/next buttons"""
    if mode==1:
        st.session_state.imgs_ix = min(len(st.session_state.imgs), st.session_state.imgs_ix+1)
    elif mode==-1:
        st.session_state.imgs_ix = max(-1, st.session_state.imgs_ix-1)

def save_results():
    """combine all generated images in a single zip file for saving"""
    zip_buf = BytesIO()
    with ZipFile(zip_buf, 'a') as file:
        for i in range(len(st.session_state.imgs)):
            _, img_buf = cv2.imencode('.png',st.session_state.imgs[i]['inp'])
            file.writestr('original_imgs/{}.png'.format(st.session_state.imgs[i]['img_name']), img_buf.tobytes())
            if st.session_state.imgs[i]['zoom'] is not None:
                _, img_zoom_buf = cv2.imencode('.png',st.session_state.imgs[i]['zoom'])
                file.writestr('zoomed_imgs/{}_zoomed.png'.format(st.session_state.imgs[i]['img_name']), img_zoom_buf.tobytes())
            if st.session_state.imgs[i]['out'] is not None:
                _, img_out_buf = cv2.imencode('.png',st.session_state.imgs[i]['out'])
                file.writestr('CBN_imgs/{}_CBN.png'.format(st.session_state.imgs[i]['img_name']), img_out_buf.tobytes())
            if st.session_state.imgs[i]['color_map'] is not None:
                _, img_map_buf = cv2.imencode('.png',st.session_state.imgs[i]['color_map'])
                file.writestr('color_maps_only/{}_cmap.png'.format(st.session_state.imgs[i]['img_name']), img_map_buf.tobytes())
            if st.session_state.imgs[i]['out_color_map'] is not None:
                _, img_out_map_buf = cv2.imencode('.png',st.session_state.imgs[i]['out_color_map'])
                file.writestr('CBN_with_color_map/{}_CBN_cmap.png'.format(st.session_state.imgs[i]['img_name']), img_out_map_buf.tobytes())
    return zip_buf

def resize(img, ix, zoom_factor):
    """Resize the image to match the target width and respect the picture ratio"""
    inp = img[ix]['inp']
    inp = cv2.resize(inp, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_AREA)
    img[ix]['zoom'] = inp

def clean(img):
    """remove noise from image"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    image =  cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, cv2.BORDER_REPLICATE)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, cv2.BORDER_REPLICATE)
    return image

def generate(img, ix, num_clrs, progress_bar):
    """generate 'color by number' image from the passed image, index and number of color clusters"""
    inp = img[ix]['zoom'] #resize is performed separately for interface convenience
    #image = inp #image = resize(inp)
    image = clean(inp)
    colormap = img[ix]['colormap_tuples']
    colortxt = img[ix]['colormap_text']
    quantized = quantize(image, colormap, colortxt)
    cbn = CBN(quantized, colormap, progress_bar)
    map_img, fontscale = draw_colormap(colormap, colortxt, inp.shape[0], inp.shape[1])
    img[ix]['quantized'] = quantized
    img[ix]['out'] = cbn
    img[ix]['color_map'] = map_img
    img[ix]['out_color_map'] = np.vstack((cbn,map_img))

def blank_colormap(img, ix, num_clrs, tolerance):
    """generate list of distinct colors in an image"""
    inp = deepcopy(img[ix]['zoom'])
    image = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    colors, pixel_count = extcolors.extract_from_image(image, tolerance=tolerance, limit=num_clrs)
    colors = [i[::-1] for (i,j) in colors] #palette of distinct colors: list of (B,G,R) tuples
    img[ix]['colormap_tuples'] = colors
    img[ix]['colormap_text'] = ['{}'.format(c[::-1]) for c in colors]

def quantize(img, colormap, colortext):
    """redraw image using colors palette"""
    image = img.reshape(img.shape[0],img.shape[1],1,3)
    colors_container = np.ones(shape=[image.shape[0],image.shape[1],len(colormap),3])
    colors = [list(c) for c in colormap] #palette of distinct colors list of lists
    for i,color in enumerate(colors):
        colors_container[:,:,i,:] = color
    shape = image.shape[:2]
    total_shape = shape[0]*shape[1]
    distances = np.sqrt(np.sum((colors_container-image)**2,axis=3))
    min_index = np.argmin(distances,axis=2).reshape(-1)
    natural_index = np.arange(total_shape)
    reshaped_container = colors_container.reshape(-1,len(colors),3)
    color_view = reshaped_container[natural_index,min_index].reshape(shape[0],shape[1],3)
    color_view = color_view.astype('uint8')
    return color_view

def CBN(img, colors, progress_bar):
    canvas = np.ones((img.shape[0],img.shape[1],img.shape[2]),dtype='uint8') * 255 #used to draw the final CBN image
    #used to draw a negative (black) of the contour to exclude areas not suitable to place text of next contour.
    negative = np.ones((img.shape[0],img.shape[1]),dtype='uint8') * 255
    #release contours from its hierarchy and have it as an unnested list of contours
    contours = []
    for ind, color in enumerate(colors):
        color = np.asarray(color, dtype='uint8')
        mask = cv2.inRange(img, color, color)
        #mask = clean(img)
        cnts,hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = [{'cnt':cnt,'ind':ind+1} for cnt in cnts if
            cv2.boundingRect(cnt)[2]>10
            and cv2.boundingRect(cnt)[3]>10
            and cv2.contourArea(cnt,False)>100]
        contours.extend(c)
    contours = sorted(contours, key= lambda x:cv2.contourArea(x['cnt'],False), reverse=False) #arcLength can also be used
    txts = tuple([str(x['ind']) for x in contours])
    contours = tuple([x['cnt'] for x in contours])

    for i, cnt in enumerate(contours):
        cv2.drawContours(canvas,[cnt],-1,0,thickness=1)

        #identify suitable place to put text
        cnt_x, cnt_y, cnt_w, cnt_h = cv2.boundingRect(cnt)
        #patch = negative[cnt_y:cnt_y+cnt_h, cnt_x:cnt_x+cnt_w, :] #get a patch from the negative
        patch = negative[cnt_y:cnt_y+cnt_h, cnt_x:cnt_x+cnt_w] #get a patch from the negative
        font_scale=1
        flag = True
        while flag:
            if font_scale >0.5: #try to find a suitable place to put the text with font scale from 1 to 0.5
                txt_w, txt_h = cv2.getTextSize(txts[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0] #get the text size in w x h
                if patch.shape[0]>txt_h and patch.shape[1]>txt_w: #check patch is bigger than text

                    #select indices that when considered as a top-left coordinate for text result in complete white box that is inside the contour
                    white_patches = np.argwhere(np.lib.stride_tricks.sliding_window_view(patch,(txt_h,txt_w)).all(axis=(-2,-1)))
                    white_patches = white_patches.tolist()
                    white_patches = [x for x in white_patches if
                                    cv2.pointPolygonTest(cnt, (x[1]+cnt_x,x[0]+cnt_y), False)>0 #TL of text in contour
                                    and cv2.pointPolygonTest(cnt, (x[1]+cnt_x+txt_w,x[0]+cnt_y), False)>0 #TR of text in contour
                                    and cv2.pointPolygonTest(cnt, (x[1]+cnt_x+txt_w,x[0]+cnt_y+txt_h), False)>0 #BR of text in contour
                                    and cv2.pointPolygonTest(cnt, (x[1]+cnt_x,x[0]+cnt_y+txt_h), False)>0 ] #BL of text in contour

                    if len(white_patches)>0: # if there are top-left coordinates found, use the first coordinate (any one can be as good) to place text
                        txt_x = white_patches[0][1]+cnt_x
                        txt_y = white_patches[0][0]+cnt_y+txt_h
                        cv2.putText(canvas, txts[i], (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, 1)
                        flag = False
                    else: #no top-left coordinates found, decrease font scale and try again
                        font_scale -=0.1
                else: #patch is smaller than text, decrease font and try again
                    font_scale -=0.1
            else: #we reached minimum possible font size. Place text at centroid of contour
                M = cv2.moments(cnt) #use contour centroid
                txt_x = int(M["m10"] / M['m00'])
                txt_y = int(M["m01"] / M['m00'])
                cv2.putText(canvas, txts[i], (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
                flag= False

        cv2.drawContours(negative,[cnt],-1,0,thickness=cv2.FILLED) #fill the contour in negative with black (to create negative)
        progress_bar.progress(i/len(contours))
    return canvas

def draw_colormap(colormap, colortext, h, w):
    """Plot or save the colormap as a picture for the user"""
    num_colors = len(colormap)
    num_patches = 4 if num_colors>=4 else num_colors #number of patches in a row
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thick = 3
    clr_txt = ['{}: {}'.format(i+1,c) for i,c in enumerate(colortext)]
    flag = True
    max_txt, max_clr, big_space, small_space = [],[], [],[]
    while flag:
        txt_sizes = [cv2.getTextSize(t, font_face, font_scale, font_thick)[0] for t in clr_txt]
        max_txt = [max(txt_sizes)[0], max(txt_sizes)[1]+10] #list wxh
        max_clr = [int(0.2 * max_txt[0])+10, int(max_txt[1])] #10 px added for 5px thick surrounding black rectangle
        ini_patch = [max_txt[0]+max_clr[0]+40, max_clr[1]] #50px added as 10 white space out_left, out_right, in_left, in_right
        full_row = num_patches * ini_patch[0]
        if w // full_row == 1 and w % full_row >0:
            #optimum size, arrange accordingly
            white_space = w - full_row
            big_space = int(white_space/num_patches) #between patches
            small_space = int(big_space/2) #at start of row and end of row
            big_space = [big_space, max_clr[1]]
            small_space = [small_space, max_clr[1]]
            flag = False
        elif w // full_row > 1:
            #txt too small, need to enlarge
            font_scale += 0.1
        elif w // full_row == 0:
            #txt too big, need to shrink
            if w < 1000 and num_patches>2:
                num_patches -= 1
            else:
                font_scale -= 0.1
    clr_offset = [int((max_clr[0] - (0.2 * max_txt[0]))/2), int((max_clr[1]-max_txt[1])/2)]
    num_rows = (num_colors // num_patches) + (1 if num_colors % num_patches !=0 else 0)
    small = np.ones((small_space[1], small_space[0], 3), dtype='uint8')*255
    big = np.ones((big_space[1], big_space[0], 3), dtype='uint8')*255
    mapbuild = None
    for row in range(num_rows):
        for col in range(num_patches):
            ix = (row*num_patches)+col
            clr = np.ones((max_clr[1], max_clr[0], 3), dtype='uint8')*255
            txt = np.ones((max_txt[1], max_txt[0], 3), dtype='uint8')*255
            if ix < len(colormap):
                clr = cv2.rectangle(clr, (0+clr_offset[0],0+clr_offset[1]), (max_clr[0]-clr_offset[0],max_clr[1]-clr_offset[1]), colormap[ix], thickness=-1)
                clr = cv2.rectangle(clr, (0+clr_offset[0],0+clr_offset[1]), (max_clr[0]-clr_offset[0],max_clr[1]-clr_offset[1]), (0,0,0), thickness=3)
                txt = cv2.putText(txt, clr_txt[ix], (0,max_txt[1]-5), font_face, font_scale, (0,0,0), thickness=1)
            if col == 0:
                rowbuild = np.hstack((small,clr,txt))
            elif col == num_patches-1:
                rowbuild = np.hstack((rowbuild,big,clr,txt,small))
            else:
                rowbuild = np.hstack((rowbuild,big,clr,txt))
        v_space = np.ones((5, rowbuild.shape[1], 3), dtype='uint8')*255
        if mapbuild is None:
            mapbuild = np.vstack((v_space,rowbuild, v_space))
        else:
            mapbuild = np.vstack((mapbuild, rowbuild, v_space))
    pad = w - mapbuild.shape[1]
    pad = np.ones((mapbuild.shape[0],pad,3), dtype='uint8')*255
    mapbuild = np.hstack((mapbuild,pad))
    return mapbuild, font_scale

def main():
    # defining side bar settings for easier access
    is_clear = st.sidebar.button('Clear All Data')
    if is_clear:
        st.sidebar.info('To properly delete all data, make sure there are no files listed by name below the file uploader window by pressing the [X] button beside each file')
        for key in st.session_state.keys():
            del st.session_state[key]

    # defining session state variables
    img_dict = {'img_name':'', 'inp':'', 'zoom':'', 'out':None, 'quantized': None, 'color_map': None, 'out_color_map':None, 'colormap_tuples':[], 'colormap_text':[]}
    if 'imgs' not in st.session_state.keys():
        st.session_state.imgs=[]
    if 'imgs_ix' not in st.session_state.keys():
        st.session_state.imgs_ix = -1
    if 'tmp_file_set' not in st.session_state.keys():
        st.session_state.tmp_file_set = set()

    # Main application
    st.title('Color By Numbers')

    # read images from file system and store them in session_state.imgs
    img_streams = st.file_uploader('Select Images', type=['png','jpeg','jpg'],  accept_multiple_files=True)
    if len(img_streams)>0:
        new_file_names = [i.name for i in img_streams]
        new_file_set = set(new_file_names)
        diff_file_set = new_file_set.difference(st.session_state.tmp_file_set)
        for img_stream in img_streams:
            if img_stream.name in diff_file_set:
                img_bytes = img_stream.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                entry = deepcopy(img_dict)
                entry['img_name'] = img_stream.name
                entry['inp'] = cv_img
                entry['zoom'] = cv_img
                st.session_state.imgs.append(entry)
                st.session_state.tmp_file_set.add(img_stream.name)
                st.session_state.imgs_ix +=1

    # add navigation controls
    col01, col02, col03, col04, col05 = st.columns([2,1,2,2,2]) #prev, imgs_ix, next, generate, save_all
    if st.session_state.imgs_ix>=0:
        progress = st.progress(0)
        zoom_factor = st.select_slider('Zoom', options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3], value=1)
        resize(st.session_state.imgs, st.session_state.imgs_ix, zoom_factor)
        cntnr01 = st.expander('ColorMap Controls', expanded=True)
        with cntnr01:
            num_clrs_slider = st.slider('Select number of colors required', min_value=1, max_value=50, value=10, step=int(1))
            tolerance_slider = st.slider('Select tolerance to group similar colors', min_value=0, max_value=100, value=32, step=int(1))
            _,col101, col102,_ = st.columns([1,1,1,1]) # 101: color map button, 102: update caption button

            col201, col202 = st.columns([1,2]) #201: color patch, 202: caption
            if len(st.session_state.imgs[st.session_state.imgs_ix]['colormap_tuples'])>0:
                txt_lst = ['']* len(st.session_state.imgs[st.session_state.imgs_ix]['colormap_text'])
                with col201:
                    for color in st.session_state.imgs[st.session_state.imgs_ix]['colormap_tuples']:
                        st.write('')
                        st.write('')
                        st.image(cv2.rectangle(np.zeros((40,40,3),dtype='uint8'),(2,2),(38,38),color, thickness=-1), channels='BGR')
                with col202:
                    for i, text in enumerate(st.session_state.imgs[st.session_state.imgs_ix]['colormap_text']):
                        placeholder ='{}'.format(st.session_state.imgs[st.session_state.imgs_ix]['colormap_tuples'][i][::-1])
                        key = '{}-{}'.format(st.session_state.imgs[st.session_state.imgs_ix]['img_name'],i)
                        txt_lst[i] = st.text_input('Caption '+str(i),value = text, placeholder=placeholder, key=key)
                        if len(txt_lst[i])==0:
                            txt_lst[i] = placeholder
            with col101:
                blank_clr_btn = st.button('ColorMap', on_click=blank_colormap, args=(st.session_state.imgs, st.session_state.imgs_ix, num_clrs_slider, tolerance_slider))
            with col102:
                update_caption_dsbl = True if len(st.session_state.imgs[st.session_state.imgs_ix]['colormap_tuples']) == 0 else False
                is_update_caption = st.button('Update Caption', disabled=update_caption_dsbl)
                if is_update_caption:
                    st.session_state.imgs[st.session_state.imgs_ix]['colormap_text'] = txt_lst
    else:
        num_clrs_slider = 0
        zoom_factor = 1
        tolerance_slider=32

    with col01:
        prev_dsbl = True if st.session_state.imgs_ix <=0 else False
        is_prev = st.button('Prev', on_click=ix_change, kwargs={'mode':-1}, disabled=prev_dsbl)
    with col02:
        st.write(str(st.session_state.imgs_ix) if st.session_state.imgs_ix>=0 else '-')
    with col03:
        next_dsbl = True if st.session_state.imgs_ix >= len(st.session_state.imgs)-1 else False
        is_next = st.button('Next', on_click=ix_change, kwargs={'mode':1}, disabled=next_dsbl)
    with col04:
        gen_dsbl = True if len(st.session_state.imgs)<=0 else False
        is_gen = st.button('Generate', on_click=generate, args=(st.session_state.imgs, st.session_state.imgs_ix, num_clrs_slider, progress), disabled=gen_dsbl)
        if is_gen:
            st.balloons()
    with col05:
        save_dsbl = True if len(st.session_state.imgs)<=0 else False
        dt = '%Y-%m-%d-%H-%M-%S'
        is_save_all = st.download_button('Save All', save_results(), file_name='ColorByNumber-'+datetime.now().strftime(dt)+'.zip', mime='application/x-zip', disabled=save_dsbl)

    col11, col12 = st.columns(2)
    with col11:
        # display the input image
        if st.session_state.imgs_ix>=0:
            st.image(st.session_state.imgs[st.session_state.imgs_ix]['zoom'], channels='BGR')
            h,w = st.session_state.imgs[st.session_state.imgs_ix]['zoom'].shape[:2]
            st.write('Resolution: {} x {}'.format(w,h))
    with col12:
        # display the quantized image
        if st.session_state.imgs_ix>=0 and st.session_state.imgs[st.session_state.imgs_ix]['quantized'] is not None:
            st.image(st.session_state.imgs[st.session_state.imgs_ix]['quantized'], channels='BGR')
            st.write('Image redrawn using ColorMap')

    col21, col22 = st.columns(2)
    with col21:
        # display the CBN image
        if st.session_state.imgs_ix>=0 and st.session_state.imgs[st.session_state.imgs_ix]['out'] is not None:
            st.image(st.session_state.imgs[st.session_state.imgs_ix]['out'], channels='BGR')
            st.write('Color By Number Image')
    with col22:
        # display the output image + color_map combined
        if st.session_state.imgs_ix>=0 and st.session_state.imgs[st.session_state.imgs_ix]['out_color_map'] is not None:
            st.image(st.session_state.imgs[st.session_state.imgs_ix]['out_color_map'], channels='BGR')
            st.write('Final Image with ColorMap')
    # display the color map
    if st.session_state.imgs_ix>=0 and st.session_state.imgs[st.session_state.imgs_ix]['color_map'] is not None:
        st.image(st.session_state.imgs[st.session_state.imgs_ix]['color_map'], channels='BGR')
        st.write('ColorMap')

if __name__ == '__main__':
    main()
