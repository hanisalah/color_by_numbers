import cv2
import numpy as np
import streamlit as st
from io import BytesIO
from copy import deepcopy
from zipfile import ZipFile
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

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

def recreate_image(cookbook, labels, w, h):
    vfunc = lambda x: cookbook[labels[x]]
    image = vfunc(np.arange(h*w))
    image = np.resize(image, (h, w, cookbook.shape[1]))
    return image

def quantize(img, num_clrs):
    H, W, D = img.shape[:3]
    image = np.reshape(img, (W*H, D))
    image_array_sample = shuffle(image, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=num_clrs).fit(image_array_sample)
    labels = kmeans.predict(image)
    image = recreate_image(kmeans.cluster_centers_, labels, W, H)
    return image, kmeans.cluster_centers_

def generate(img, ix, num_clrs):
    """generate 'color by number' image from the passed image, index and number of color clusters"""
    inp = img[ix]['zoom'] #resize is performed separately for interface convenience
    #image = inp #image = resize(inp)
    image = clean(inp)
    image = np.array(image, dtype='uint8')/255
    image, colors = quantize(image, num_clrs)
    canvas = np.ones(image.shape[:3], dtype="uint8")*255
    colormap = [[int(c*255) for c in col] for col in colors]
    h, w = canvas.shape[:2]
    map_img, font_scale = draw_colormap(colormap, h, w)
    for ind, color in enumerate(colors):
        #colormap.append([int(c*255) for c in color])
        mask = cv2.inRange(image, color, color)
        cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for contour in cnts:
            _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
            if width_ctr > 10 and height_ctr > 10 and cv2.contourArea(contour, True) < -100:
                cv2.drawContours(canvas, [contour], -1, 0, 1)
                #Add label
                #txt_x, txt_y = contour[0][0]
                M = cv2.moments(contour) #use contour centroid
                txt_x = int(M["m10"] / M["m00"])
                txt_y = int(M["m01"] / M["m00"])
                cv2.putText(canvas, '{:d}'.format(ind + 1), (txt_x, txt_y + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, 1) #originally font scale = 0.5
    img[ix]['out'] = canvas
    img[ix]['color_map'] = map_img
    img[ix]['out_color_map'] = np.vstack((canvas,map_img))

def draw_colormap(colormap, h, w):
    """Plot or save the colormap as a picture for the user"""
    num_colors = len(colormap)
    num_patches = 4 if num_colors>=4 else num_colors #number of patches in a row
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thick = 3
    clr_txt = ['{}: {}'.format(i+1,c[::-1]) for i,c in enumerate(colormap)]
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
                clr = cv2.rectangle(clr, (0+clr_offset[0],0+clr_offset[1]), (max_clr[0]-clr_offset[0],max_clr[1]-clr_offset[1]), (0,0,0), thickness=1)
                txt = cv2.putText(txt, clr_txt[ix], (0,max_txt[1]-5), font_face, font_scale, (0,0,0), thickness=1)
            if col == 0:
                rowbuild = np.hstack((small,clr,txt))
            elif col == num_patches-1:
                rowbuild = np.hstack((rowbuild,big,clr,txt,small))
            else:
                rowbuild = np.hstack((rowbuild,big,clr,txt))
        if mapbuild is None:
            mapbuild = rowbuild
        else:
            mapbuild = np.vstack((mapbuild, rowbuild))
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
    img_dict = {'img_name':'', 'inp':'', 'zoom':'', 'out':None, 'color_map': None, 'out_color_map':None}
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
    col01, col02, col03, col04, col05 = st.columns([2,1,2,2,2])
    with col01:
        prev_dsbl = True if st.session_state.imgs_ix <=0 else False
        is_prev = st.button('Prev', on_click=ix_change, kwargs={'mode':-1}, disabled=prev_dsbl)
    with col02:
        st.write(str(st.session_state.imgs_ix) if st.session_state.imgs_ix>=0 else '-')
    with col03:
        next_dsbl = True if st.session_state.imgs_ix >= len(st.session_state.imgs)-1 else False
        is_next = st.button('Next', on_click=ix_change, kwargs={'mode':1}, disabled=next_dsbl)
    if st.session_state.imgs_ix>=0:
        zoom_factor = st.select_slider('Zoom', options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3], value=1)
        num_clrs_slider = st.slider('Select number of color clusters required', min_value=1, max_value=50, value=10, step=int(1))
        resize(st.session_state.imgs, st.session_state.imgs_ix, zoom_factor)
    else:
        num_clrs_slider = 0
        zoom_factor = 1
    with col04:
        gen_dsbl = True if len(st.session_state.imgs)<=0 else False
        is_gen = st.button('Generate', on_click=generate, args=(st.session_state.imgs, st.session_state.imgs_ix, num_clrs_slider), disabled=gen_dsbl)
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
        # display the output image
        if st.session_state.imgs_ix>=0 and st.session_state.imgs[st.session_state.imgs_ix]['out'] is not None:
            st.image(st.session_state.imgs[st.session_state.imgs_ix]['out'], channels='BGR')

    col21, col22 = st.columns(2)
    with col21:
        # display the color_map
        if st.session_state.imgs_ix>=0 and st.session_state.imgs[st.session_state.imgs_ix]['color_map'] is not None:
            st.image(st.session_state.imgs[st.session_state.imgs_ix]['color_map'], channels='BGR')
    with col22:
        # display the output image + color_map combined
        if st.session_state.imgs_ix>=0 and st.session_state.imgs[st.session_state.imgs_ix]['out_color_map'] is not None:
            st.image(st.session_state.imgs[st.session_state.imgs_ix]['out_color_map'], channels='BGR')

if __name__ == '__main__':
    main()
