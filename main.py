import cv2
import ssl
import json
import time
import hashlib
import certifi
import extcolors
import numpy as np
from PIL import Image
import mysql.connector
import streamlit as st
from io import BytesIO
import cryptocode as crc
from copy import deepcopy
from zipfile import ZipFile
from datetime import datetime
import largestinteriorrectangle as lir
from urllib import request, parse, error

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

def remove_border(img, ver, hor):
    # clean out border
    v = int(ver*img.shape[0])
    h = int(hor*img.shape[1])
    color = img[0,0].tolist() #[255,255,255]
    #equalize frame color between image border and end of frame (assumed as 5 pixels after defined frame)
    if v>0:
        img[:v+5,:] = color
        img[img.shape[0]-v-5:img.shape[0],:]=color
    if h>0:
        img[:, :h+5] = color
        img[:, img.shape[1]-h-5:img.shape[1]]=color
    #flood fill the equalized frame color with background color
    if v>0 or h>0:
        #remove background
        img = cv2.floodFill(img, None, (0,0), color)[1]
        img = cv2.floodFill(img, None, (img.shape[1]-5,0), color)[1]
        img = cv2.floodFill(img, None, (0,img.shape[0]-5), color)[1]
        img = cv2.floodFill(img, None, (img.shape[1]-5,img.shape[0]-5), color)[1]

    # add 1% pixels to image with same color in each direction to isolate outer most contour (it will be cleared in CBN)
    hor = np.ones((img.shape[0], max(int(img.shape[1]*0.01),15), img.shape[2]), dtype='uint8')
    hor[...] = [255,255,255]
    img = np.hstack((hor,img,hor))
    ver = np.ones((max(int(img.shape[0]*0.01),15), img.shape[1], img.shape[2]), dtype='uint8')
    ver[...] = [255,255,255]
    img = np.vstack((ver,img,ver))

    return img

def clean(img):
    """remove noise from image"""
    #denoise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    image =  cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return image

def generate(img, ix, num_clrs, progress_bar):
    """generate 'color by number' image from the passed image, index and number of color clusters"""
    inp = deepcopy(img[ix]['zoom']) #resize is performed separately for interface convenience
    image = clean(inp)
    colormap = img[ix]['colormap_tuples']
    colortxt = img[ix]['colormap_text']
    quantized = quantize(image, colormap, colortxt)
    cbn = CBN(quantized, colormap, progress_bar)
    map_img, fontscale = draw_colormap(colormap, colortxt, inp.shape[0], inp.shape[1], img[ix]['show_clr_box'])
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
    """ Draw Color-By-Numbers Image """
    canvas = np.ones((img.shape[0],img.shape[1],img.shape[2]),dtype='uint8') * 255 #used to draw the final CBN image

    #release contours from its hierarchy and have it as an unnested list of contours
    contours = []
    for ind, color in enumerate(colors):
        color = np.asarray(color, dtype='uint8')
        mask = cv2.inRange(img, color, color)
        cnts,hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        c = [{'cnt':cnt,'ind':ind+1} for cnt in cnts if
            cv2.boundingRect(cnt)[2]>10
            and cv2.boundingRect(cnt)[3]>10
            and cv2.contourArea(cnt,False)>100]
        contours.extend(c)
    contours = sorted(contours, key= lambda x:cv2.contourArea(x['cnt'],False), reverse=False) #arcLength, contourArea can also be used
    txts = tuple([str(x['ind']) for x in contours])
    contours = tuple([x['cnt'] for x in contours])

    passed_cnts = []

    for i, cnt in enumerate(contours):
        # draw contour
        cv2.drawContours(canvas,contours,i,0,thickness=1)

        # determine best location and best text size to place text
        font_scale = 1
        flag =True

        #used to draw a negative (black) of the contour to exclude areas not suitable to place text of next contour.
        negative = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
        cv2.drawContours(negative, [cnt], -1, 1, thickness=cv2.FILLED) #draw contour filled with value 1 (indicative of white for uint8 img, indicative of true for bool img)
        if len(passed_cnts)!=0: # if this is not the first contour
            for p_cnt in passed_cnts:
                cv2.drawContours(negative, [p_cnt], -1, 0, thickness=cv2.FILLED) #draw all previous contours filled with value 0 (indicative of black for uint8 img, false for bool img)
        negative = negative.astype('bool') #convert negative to bool array
        c = cnt[:,0,:] # prepare contour
        x,y,w,h = lir.lir(negative, c) #get largest inerior white rectangle within contour c
        while flag and font_scale >0.1:
            txt_w, txt_h = cv2.getTextSize(txts[i], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0] #get the text size in w x h
            if w>= txt_w and h >= txt_h: #if largest white interior rectangle is bigger than text dimensions
                txt_x = x + int((w - txt_w)/2)
                txt_y = y + txt_h + int((h - txt_h)/2)
                cv2.putText(canvas, txts[i], (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 0, 1) #place text
                negative = negative.astype('uint8') #convert negative to uint8
                cv2.drawContours(negative,contours,i,0,thickness=cv2.FILLED) #fill the contour in negative with black (to create negative)
                negative = negative.astype('bool') #convert negative to bool to be used again in lir
                passed_cnts.append(cnt) #append contour to list of passed contours
                flag = False #exit while loop for this contour
            else: #largest white interior rectangle is smaller than text dimension, reduce font scale and try again
                font_scale -= 0.1
        progress_bar.progress(i/len(contours))

    #Now we need to remove the outermost contour representing a frame for the background of the image
    negative = np.zeros((img.shape[0],img.shape[1]),dtype='uint8') #total black image
    for p_cnt in passed_cnts:
        cv2.drawContours(negative, [p_cnt], -1, 255) #draw all contors as white lines on black image
    cnts, _ = cv2.findContours(negative, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #get the outermost contour
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas,[cnt],-1,(255,255,255)) #draw the outermost contour in white over the canvas (our result image)

    return canvas

def draw_colormap(colormap, colortext, h, w, show_clr_box):
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
                if show_clr_box:
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

#@st.experimental_singleton caches the function to execute only once. While good to avoid multiple connections, but sometimes the connection is lost.
def init_connection():
    """Initiate connection to database. """
    return mysql.connector.connect(**st.secrets['mysql'])

def query_db(query):
    """Execute SQL query on database and returns number of rows, rows content. """
    conn = init_connection()
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    return len(rows), rows

def licenseCheck(license_key, product_id):
    url_ = "https://api.gumroad.com/v2/licenses/verify"
    pf_permalink = product_id
    params = {'product_permalink': pf_permalink, 'license_key': license_key}
    data=parse.urlencode(params).encode('ascii')
    req = request.Request(url_, data=data)
    try:
        response = request.urlopen(req, context=ssl.create_default_context(cafile=certifi.where()))
        get_response = json.loads(response.read())
    except error.HTTPError as e:
        get_response = json.loads(e.read())
    status = False
    if get_response['success']:
        status = True
    else:
        get_response = "Failed to verify the license."
    return status, get_response , license_key

def signUp(name, email, username, password, rep_password, license_key):
    st.session_state.user = ''
    st.session_state.logged = False
    msg = []
    msg_state = []
    flag = True
    num = 0
    rows = []
    username = username.lower()

        # from db each line is [id, name_pass, email_pass, username_pass, hash_license, hash_user, hash_email]
    if password != rep_password: #password mismatch
        msg.append('Password and Repeat Password fields are not the same. Please Try Again!')
        msg_state.append('warning')
        flag = False
    if flag:
        if licenseCheck(license_key,'wkdhw')[0] != True: #license key is not correct
            msg.append('License Key is incorrect!')
            msg_state.append('error')
            flag = False
    if flag:
        num, rows = query_db("SELECT HL, HU, HE From users WHERE HL = '{}' OR HU = '{}' OR HE = '{}'".format(
                            hashlib.sha512(license_key.encode()).hexdigest(), hashlib.sha512(username.encode()).hexdigest(), hashlib.sha512(email.encode()).hexdigest() ))
        if num > 0:
            if hashlib.sha512(license_key.encode()).hexdigest() in [r[0] for r in rows]: #license is used already
                msg.append('This license is already registered!')
                msg_state.append('error')
                flag = False
            if hashlib.sha512(username.encode()).hexdigest() in [r[1] for r in rows]: #username exists
                msg.append('Username is already used. Please use another Username')
                msg_state.append('warning')
                flag = False
            if hashlib.sha512(email.encode()).hexdigest() in [r[2] for r in rows]: #email exists
                msg.append('This email is already used.')
                msg_state.append('warning')
                flag = False

    if flag: #all checks passed
        NP = crc.encrypt(name, password)
        EP = crc.encrypt(email, password)
        UP = crc.encrypt(username, password)
        HL = hashlib.sha512(license_key.encode()).hexdigest()
        HU = hashlib.sha512(username.encode()).hexdigest()
        HE = hashlib.sha512(email.encode()).hexdigest()
        query = """ INSERT INTO users (NP, EP, UP, HL, HU, HE)
                    VALUES ('{}', '{}', '{}', '{}', '{}', '{}')""".format(NP, EP, UP, HL, HU, HE)
        query_db(query)

        st.session_state.user = name
        st.session_state.logged = True
        msg.append('Registration Successful')
        msg_state.append('success')
    return msg_state, msg

def signIn(username, password, isGuest):
    st.session_state.user = ''
    st.session_state.logged = False
    msg = []
    msg_state = []
    username = username.lower()

    if isGuest == False:
        # from db each line is [id, name_pass, email_pass, username_pass, hash_password, hash_license, hash_user, hash_email]
        num, rows = query_db("SELECT NP, UP FROM users WHERE HU = '{}' ".format( hashlib.sha512(username.encode()).hexdigest() ))

        if num == 1 and crc.decrypt(rows[0][1], password) == username:
            st.session_state.user = crc.decrypt(rows[0][0], password)
            st.session_state.logged = True
            msg.append('Sign In Successful')
            msg_state.append('success')
        else:
            msg.append('Username / Password Mismatch! Please try again.')
            msg_state.append('error')
    else:
        st.session_state.user = 'Guest'
        st.session_state.logged = True
        msg.append('Sign In Successful')
        msg_state.append('success')
    return msg_state, msg

def forgotDetails(license_key, email):
    st.session_state.user = ''
    st.session_state.logged = False
    msg = []
    msg_state = []

    num, rows = query_db("SELECT * FROM users WHERE HL = '{}' AND HE = '{}' ".format(
                        hashlib.sha512(license_key.encode()).hexdigest(), hashlib.sha512(email.encode()).hexdigest() ))

    if num == 0:
        msg.append('This License Key / email combination is not available in the system. Please make sure the license key and/or email is correct and/or proceed to Sign Up')
        msg_state.append('error')

    if num == 1:
        query_db("DELETE FROM users WHERE HL = '{}' AND HE = '{}' ".format(
                    hashlib.sha512(license_key.encode()).hexdigest(), hashlib.sha512(email.encode()).hexdigest() ))
        msg.append('Your Data has been removed from system. Please Sign Up Again')
        msg_state.append('success')

    return msg_state, msg

def topBar():
    st.title('Color By Numbers')
    msg_container = st.empty()
    signin_tab, forgot_tab, signup_tab = st.tabs(['Sign In', 'Forgot Details', 'Sign Up'])
    signin_form, signforget_form, signup_form = None, None, None
    msg_states=[]

    with signup_tab:
        signup_form = st.form('signUp-form', clear_on_submit=True)
        signup_form.subheader('Register New User')
        signup_name = str(signup_form.text_input('Name', value=''))
        signup_email = str(signup_form.text_input('email', value=''))
        signup_username = str(signup_form.text_input('Enter Username', value=''))
        signup_password = str(signup_form.text_input('Enter Password', value='', type='password'))
        signup_rep_password = str(signup_form.text_input('Enter Password again', value='', type='password'))
        signup_license_key = str(signup_form.text_input('Enter your License Key', value=''))
        signUp_Btn = signup_form.form_submit_button('Sign Up')
    if signUp_Btn:
        msg_states, msgs = signUp(signup_name, signup_email, signup_username, signup_password, signup_rep_password, signup_license_key)

    with signin_tab:
        signin_form = st.form('signIn-form', clear_on_submit=True)
        signin_form.subheader('Sign In')
        signIn_username = str(signin_form.text_input('Username', value=''))
        signIn_password = str(signin_form.text_input('Password', value='', type='password'))
        signIn_guest = signin_form.checkbox('Guest Sign In')
        signIn_Btn = signin_form.form_submit_button('Sign In')
    if signIn_Btn:
        msg_states, msgs = signIn(signIn_username, signIn_password, signIn_guest)

    with forgot_tab:
        signforget_form = st.form('ForgotCredentials-form', clear_on_submit=True)
        signforget_form.subheader('Reset Your Data')
        forgot_email = str(signforget_form.text_input('email', value=''))
        forgot_license = str(signforget_form.text_input('License Key', value=''))
        forgot_Btn = signforget_form.form_submit_button('Reset Data')
    if forgot_Btn:
        msg_states, msgs = forgotDetails(forgot_license, forgot_email)

    with msg_container.container():
        if len(msg_states)>0:
            for i,msg_state in enumerate(msg_states):
                if msg_state == 'success':
                    st.success(msgs[i])
                elif msg_state == 'warning':
                    st.warning(msgs[i])
                elif msg_state == 'error':
                    st.error(msgs[i])
            time.sleep(3)
            msg_states=[]
            msgs = []
            msg_container.empty()

def main():
    # defining session state variables
    if 'user' not in st.session_state.keys():
        st.session_state.user = ''
    if 'logged' not in st.session_state.keys():
        st.session_state.logged = False

    img_dict = {'img_name':'', 'inp':'', 'zoom':'',
                'out':None, 'quantized': None, 'color_map': None,
                'out_color_map':None, 'colormap_tuples':[], 'colormap_text':[],
                'show_clr_box':True, 'zoom_factor':1, 'hor_border':0.0, 'ver_border':0.0}
    if 'imgs' not in st.session_state.keys():
        st.session_state.imgs=[]
    if 'imgs_ix' not in st.session_state.keys():
        st.session_state.imgs_ix = -1
    if 'tmp_file_set' not in st.session_state.keys():
        st.session_state.tmp_file_set = set()

    # login panel
    if st.session_state.logged == False:
        topBar()
        time.sleep(2)
        if st.session_state.logged == True:
            st.experimental_rerun()

    # Main application
    if st.session_state.logged:
        # defining side bar settings for easier access
        is_logout = st.sidebar.button('Sign Out')
        if is_logout:
            for key in st.session_state.keys():
                del st.session_state[key]
            st.session_state.user = ''
            st.session_state.logged = False
            st.experimental_rerun()
        is_clear = st.sidebar.button('Clear All Data')
        if is_clear:
            st.sidebar.info('To properly delete all data, make sure there are no files listed by name below the file uploader window by pressing the [X] button beside each file')
            time.sleep(3)
            for key in st.session_state.keys():
                if key not in ['user','logged', 'state']:
                    del st.session_state[key]
            st.experimental_rerun()

        # Main application
        cols = st.columns(5)
        with cols[-1]:
            st.write('*Hello, {}*'.format(st.session_state.user))
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
                    cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    if cv_img.shape[2] == 4:
                        h,w = cv_img.shape[0], cv_img.shape[1]
                        bg = (255,255,255)
                        bgr = np.zeros((h,w,3), dtype='float32')
                        b, g, r, a = cv2.split(cv_img)
                        a = np.asarray(a, dtype='float32')/255.0
                        R,G,B = bg
                        bgr[:,:,0] = b*a + (1.0 - a) * B
                        bgr[:,:,1] = g*a + (1.0 - a) * G
                        bgr[:,:,2] = r*a + (1.0 - a) * R
                        cv_img = np.asarray(bgr, dtype='uint8')

                    entry = deepcopy(img_dict)
                    entry['img_name'] = img_stream.name
                    entry['inp'] = cv_img
                    entry['zoom'] = cv_img
                    st.session_state.imgs.append(entry)
                    st.session_state.tmp_file_set.add(img_stream.name)
                    st.session_state.imgs_ix +=1

        # add navigation controls
        col01, col02, col03, col04, col05 = st.columns([2,1,2,2,2]) #prev, imgs_ix, next, generate, save_all
        progress = st.progress(0)
        if st.session_state.imgs_ix>=0:
            cntnr01 = st.expander('ColorMap Controls', expanded=False)
            cntnr02 = st.expander('Image Controls', expanded=False)

            with cntnr02:
                key= st.session_state.imgs[st.session_state.imgs_ix]['img_name']
                zoom_factor = st.select_slider('Zoom', options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3], value=st.session_state.imgs[st.session_state.imgs_ix]['zoom_factor'], key=key+'z')
                st.session_state.imgs[st.session_state.imgs_ix]['zoom_factor'] = zoom_factor
                resize(st.session_state.imgs, st.session_state.imgs_ix, zoom_factor)
                col31, col32 = st.columns(2)
                with col31:
                    hor = st.slider('Horizontal Border Limit', min_value=0, max_value=15, value=int(st.session_state.imgs[st.session_state.imgs_ix]['hor_border'] * 100), step=1, key=key+'h') / 100
                    st.session_state.imgs[st.session_state.imgs_ix]['hor_border'] = hor
                with col32:
                    ver = st.slider('Vertical Border Limit', min_value=0, max_value=15, value=int(st.session_state.imgs[st.session_state.imgs_ix]['ver_border'] *100), step=1, key=key+'v') / 100
                    st.session_state.imgs[st.session_state.imgs_ix]['ver_border'] = ver
                st.session_state.imgs[st.session_state.imgs_ix]['zoom'] = remove_border(st.session_state.imgs[st.session_state.imgs_ix]['zoom'],ver,hor)

            with cntnr01:
                num_clrs_slider = st.slider('Select number of colors required', min_value=1, max_value=50, value=10, step=int(1))
                tolerance_slider = st.slider('Select tolerance to group similar colors', min_value=0, max_value=100, value=32, step=int(1))
                col101,_,col102, _,col103 = st.columns([2,1,2,1,2]) # 101: color map button, 102: update caption button

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
                    key = '{}-clrbox-chkbx'.format(st.session_state.imgs[st.session_state.imgs_ix]['img_name'])
                    st.session_state.imgs[st.session_state.imgs_ix]['show_clr_box'] = st.checkbox('Show Color Box', value=st.session_state.imgs[st.session_state.imgs_ix]['show_clr_box'], key=key)
                with col102:
                    blank_clr_btn = st.button('ColorMap', on_click=blank_colormap, args=(st.session_state.imgs, st.session_state.imgs_ix, num_clrs_slider, tolerance_slider))
                with col103:
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
            gen_dsbl = True if len(st.session_state.imgs)<=0 or len(st.session_state.imgs[st.session_state.imgs_ix]['colormap_tuples'])<=0 else False
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
                st.write('Zoomed / Padded / Cropped Image')
                #h,w = st.session_state.imgs[st.session_state.imgs_ix]['zoom'].shape[:2]
                #st.write('Zoomed / Padded / Cropped Image - Resolution: {} x {}'.format(w,h))
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
