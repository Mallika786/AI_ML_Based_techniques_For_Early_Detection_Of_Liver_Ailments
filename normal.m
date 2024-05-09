function varargout = normal(varargin)
% NORMAL MATLAB code for normal.fig
%      NORMAL, by itself, creates a new NORMAL or raises the existing
%      singleton*.
%
%      H = NORMAL returns the handle to a new NORMAL or the handle to
%      the existing singleton*.
%
%      NORMAL('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NORMAL.M with the given input arguments.
%
%      NORMAL('Property','Value',...) creates a new NORMAL or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before normal_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to normal_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help normal

% Last Modified by GUIDE v2.5 29-Feb-2024 22:38:50

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @normal_OpeningFcn, ...
                   'gui_OutputFcn',  @normal_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before normal is made visible.
function normal_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to normal (see VARARGIN)

% Choose default command line output for normal
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes normal wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = normal_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

global im1 im2
% --- Executes on button press in upload_image.
function upload_image_Callback(hObject, eventdata, handles)
% hObject    handle to upload_image (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im1 im2
[path, nofile]=imgetfile();
if nofile
    msgbox(sprintf("image not found"),'Error','Warning');
    return
end
im1=imread(path);
im1=im2double(im1);
im2=im1;
axes(handles.axes1);
imshow(im1)

function pre_processing_Callback(hObject, eventdata, handles)
% hObject    handle to pre_processing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im1;

% Check if the image is loaded
if isempty(im1)
    msgbox('Please load an image first', 'Error', 'Warning');
    return
end

% Convert the image to grayscale if it's not already
if size(im1, 3) == 3
    grayImage = rgb2gray(im1);
else
    grayImage = im1;
end

% Step 1: Noise reduction using Gaussian filtering
filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

% Step 2: Enhance contrast using histogram equalization
enhancedImage = histeq(filteredImage);

% Display the preprocessed image
axes(handles.axes2);
imshow(enhancedImage);

function edge_detection_Callback(hObject, eventdata, handles)
% hObject    handle to edge_detection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im1;

% Check if the image is loaded
if isempty(im1)
    msgbox('Please load an image first', 'Error', 'Warning');
    return
end

% Convert the image to grayscale if it's not already
if size(im1, 3) == 3
    grayImage = rgb2gray(im1);
else
    grayImage = im1;
end

% Step 1: Noise reduction using Gaussian filtering
filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

% Step 2: Enhance contrast using histogram equalization
enhancedImage = histeq(filteredImage);

% Step 1: Canny edge detection
edges = edge(enhancedImage, 'Canny');

% Display the original image and detected edges
axes(handles.axes3);
imshow(edges);
function segmentation_Callback(hObject, eventdata, handles)
% hObject    handle to segmentation (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im1;

% Check if the image is loaded
if isempty(im1)
    msgbox('Please load an image first', 'Error', 'Warning');
    return
end

% Convert the image to grayscale if it's not already
if size(im1, 3) == 3
    grayImage = rgb2gray(im1);
else
    grayImage = im1;
end

% Step 1: Noise reduction using Gaussian filtering
filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

% Step 2: Enhance contrast using histogram equalization
enhancedImage = histeq(filteredImage);

% Step 1: Canny edge detection
%edges = edge(enhancedImage, 'Canny');
a=get(handles.popupmenu1,'value');
    switch a
        case 1
            edges = edge(enhancedImage, 'Canny');
        case 2
            edges = edge(enhancedImage, 'Sobel');
        case 3
            edges = edge(enhancedImage, 'Prewitt');
        case 4
            edges = edge(enhancedImage, 'Roberts');
    end

threshold = 0.2; % Adjust as needed
binaryImage = edges > threshold;

% Perform morphological operations for further refinement of the segmentation
% Here, we perform closing to fill small gaps in the detected edges
se = strel('disk', 10); % Adjust the structuring element size as needed
binaryImage = imclose(binaryImage, se);

% Convert the binary image to a grayscale image for display
binaryImageGray = uint8(binaryImage) * 255;

% Display the segmented image
axes(handles.axes4);
imshow(binaryImageGray);

% --- Executes on button press in clear.
function clear_Callback(hObject, eventdata, handles)
% hObject    handle to clear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cla(handles.axes1, 'reset');
cla(handles.axes2, 'reset');
cla(handles.axes3, 'reset');
cla(handles.axes4, 'reset');
cla(handles.axes5, 'reset');
cla(handles.axes6, 'reset');
set(handles.edit1, 'String', '');
set(handles.edit2, 'String', '');

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in segmentation.


% --- Executes on button press in edge_detection.

% --- Executes on button press in pre_processing.


% --- Executes on button press in detection.
function detection_Callback(hObject, eventdata, handles)
% hObject    handle to detection (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im1;

% Check if the image is loaded
if isempty(im1)
    msgbox('Please load an image first', 'Error', 'Warning');
    return
end

% Convert the image to grayscale if it's not already
if size(im1, 3) == 3
    grayImage = rgb2gray(im1);
else
    grayImage = im1;
end

% Step 1: Noise reduction using Gaussian filtering
filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

% Step 2: Enhance contrast using histogram equalization
enhancedImage = histeq(filteredImage);

% Step 1: Canny edge detection
%edges = edge(enhancedImage, 'Canny');
a=get(handles.popupmenu1,'value');
    switch a
        case 1
            edges = edge(enhancedImage, 'Canny');
        case 2
            edges = edge(enhancedImage, 'Sobel');
        case 3
            edges = edge(enhancedImage, 'Prewitt');
        case 4
            edges = edge(enhancedImage, 'Roberts');
    end


    % Step 1: Noise reduction using Gaussian filtering
    

    b=get(handles.popupmenu3,'value');
    switch b
        case 1
            filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

    % Step 2: Enhance contrast using histogram equalization
            enhancedImage = histeq(filteredImage);

    % Perform edge detection based on the selected method
            a=get(handles.popupmenu1,'value');
            switch a
                case 1
                    edges = edge(enhancedImage, 'Canny');
                case 2
                    edges = edge(enhancedImage, 'Sobel');
                case 3
                    edges = edge(enhancedImage, 'Prewitt');
                case 4
                    edges = edge(enhancedImage, 'Roberts');
            end
            threshold = 0.2; % Adjust as needed
            binaryImage = edges > threshold;

% Perform morphological operations for further refinement of the segmentation
% Here, we perform closing to fill small gaps in the detected edges
            se = strel('disk', 10); % Adjust the structuring element size as needed
            binaryImage = imclose(binaryImage, se);

% Convert the binary image to a grayscale image for display
            binaryImageGray = uint8(binaryImage) * 255;
        case 2
            img = rgb2gray(im1);
            T = adaptthresh(img, 0.4);
            binaryImage = imbinarize(img,T);

            binaryImageGray = uint8(binaryImage) * 255;
        case 3
            %I = rgb2gray(im1);
%imshow(I)
%title('Original Image')
            %grayimage=rgb2gray(im1);
            mask = zeros(size(grayImage));%creates a binary mask of zeroes with the same size as grayscale image
            mask(25:end-25,25:end-25) = 1;
%imshow(mask)
            binaryImage = activecontour(grayImage,mask,500);
            binaryImage=~binaryImage;
            binaryImageGray = uint8(binaryImage) * 255;
        case 4
            I = im2gray(im1);
            gmag = imgradient(I);
            L = watershed(gmag);
            Lrgb = label2rgb(L);
            se = strel("disk",20);
            Io = imopen(I,se);
            Ie = imerode(I,se);
            Iobr = imreconstruct(Ie,I);
            Ioc = imclose(Io,se);
            Iobrd = imdilate(Iobr,se);
            Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
            Iobrcbr = imcomplement(Iobrcbr);
            fgm = imregionalmax(Iobrcbr);
            I2 = labeloverlay(I,fgm);
            se2 = strel(ones(5,5));
            fgm2 = imclose(fgm,se2);
            fgm3 = imerode(fgm2,se2);
            fgm4 = bwareaopen(fgm3,20);
            I3 = labeloverlay(I,fgm4);
            bw = imbinarize(Iobrcbr);
            D = bwdist(bw);
            DL = watershed(D);
            bgm = DL == 0;
            gmag2 = imimposemin(gmag, bgm | fgm4);
            L = watershed(gmag2);
            labels = imdilate(L==0,ones(3,3)) + 2*bgm + 3*fgm4;
            I4 = labeloverlay(I,labels);
            Lrgb = label2rgb(L,"jet","w","shuffle");
            binaryImage=rgb2gray(Lrgb);
            binaryImageGray=rgb2gray(Lrgb);
        case 5
            I=im2gray(im1);
            BW = imbinarize(I);
            [B,L] = bwboundaries(BW,'noholes');
            binaryImage=~L
            L=~L
            binaryImageGray=L
            binaryImageGray = uint8(L) * 255;
            

    end
% Convert the binary image to a grayscale image for display
% Create a mask for the background (non-segmented regions)
backgroundMask = ~binaryImageGray;

% Create a color version of the original image
colorImage = repmat(binaryImageGray, [1, 1, 3]);

% Set the background region to white
colorImage(backgroundMask > 0) = 255;

% Display the image with white background



% Create a color overlay by setting the segmented part to the overlay color
redChannel = im1(:,:,1);  % Red channel
greenChannel = im1(:,:,2);  % Green channel
blueChannel = im1(:,:,3);  % Blue channel

% Set the red channel to maximum and green and blue channels to minimum where binaryImage is true
redChannel(~binaryImage) = 255; % Set red channel to maximum where binaryImage is false (white)
greenChannel(~binaryImage) = 0; % Set green channel to minimum where binaryImage is false (white)
blueChannel(~binaryImage) = 0; % Set blue channel to minimum where binaryImage is false (white)

% Combine the channels to form the RGB image
overlayedImage = cat(3, redChannel, greenChannel, blueChannel);

% Display the original image with the segmented part overlaid
axes(handles.axes6);
imshow(overlayedImage);


% --- Executes on button press in feature_extraction.
function feature_extraction_Callback(hObject, eventdata, handles)
% hObject    handle to feature_extraction (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global im1;

% Check if the image is loaded
if isempty(im1)
    msgbox('Please load an image first', 'Error', 'Warning');
    return
end

% Convert the image to grayscale if it's not already
if size(im1, 3) == 3
    grayImage = rgb2gray(im1);
else
    grayImage = im1;
end

% Step 1: Noise reduction using Gaussian filtering
filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

% Step 2: Enhance contrast using histogram equalization
enhancedImage = histeq(filteredImage);

% Step 1: Canny edge detection
%edges = edge(enhancedImage, 'Canny');
a=get(handles.popupmenu1,'value');
    switch a
        case 1
            edges = edge(enhancedImage, 'Canny');
        case 2
            edges = edge(enhancedImage, 'Sobel');
        case 3
            edges = edge(enhancedImage, 'Prewitt');
        case 4
            edges = edge(enhancedImage, 'Roberts');
    end


    % Step 1: Noise reduction using Gaussian filtering
    

    b=get(handles.popupmenu3,'value');
    switch b
        case 1
            filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

    % Step 2: Enhance contrast using histogram equalization
            enhancedImage = histeq(filteredImage);

    % Perform edge detection based on the selected method
            a=get(handles.popupmenu1,'value');
            switch a
                case 1
                    edges = edge(enhancedImage, 'Canny');
                case 2
                    edges = edge(enhancedImage, 'Sobel');
                case 3
                    edges = edge(enhancedImage, 'Prewitt');
                case 4
                    edges = edge(enhancedImage, 'Roberts');
            end
            threshold = 0.2; % Adjust as needed
            binaryImage = edges > threshold;

% Perform morphological operations for further refinement of the segmentation
% Here, we perform closing to fill small gaps in the detected edges
            se = strel('disk', 10); % Adjust the structuring element size as needed
            binaryImage = imclose(binaryImage, se);

% Convert the binary image to a grayscale image for display
            binaryImageGray = uint8(binaryImage) * 255;
        case 2
            img = rgb2gray(im1);
            T = adaptthresh(img, 0.4);
            binaryImage = imbinarize(img,T);

            binaryImageGray = uint8(binaryImage) * 255;
        case 3
            %I = rgb2gray(im1);
%imshow(I)
%title('Original Image')
            %grayimage=rgb2gray(im1);
            mask = zeros(size(grayImage));%creates a binary mask of zeroes with the same size as grayscale image
            mask(25:end-25,25:end-25) = 1;
%imshow(mask)
            binaryImage = activecontour(grayImage,mask,500);
            binaryImage=~binaryImage;
            binaryImageGray = uint8(binaryImage) * 255;
        case 4
            I = im2gray(im1);
            gmag = imgradient(I);
            L = watershed(gmag);
            Lrgb = label2rgb(L);
            se = strel("disk",20);
            Io = imopen(I,se);
            Ie = imerode(I,se);
            Iobr = imreconstruct(Ie,I);
            Ioc = imclose(Io,se);
            Iobrd = imdilate(Iobr,se);
            Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
            Iobrcbr = imcomplement(Iobrcbr);
            fgm = imregionalmax(Iobrcbr);
            I2 = labeloverlay(I,fgm);
            se2 = strel(ones(5,5));
            fgm2 = imclose(fgm,se2);
            fgm3 = imerode(fgm2,se2);
            fgm4 = bwareaopen(fgm3,20);
            I3 = labeloverlay(I,fgm4);
            bw = imbinarize(Iobrcbr);
            D = bwdist(bw);
            DL = watershed(D);
            bgm = DL == 0;
            gmag2 = imimposemin(gmag, bgm | fgm4);
            L = watershed(gmag2);
            labels = imdilate(L==0,ones(3,3)) + 2*bgm + 3*fgm4;
            I4 = labeloverlay(I,labels);
            Lrgb = label2rgb(L,"jet","w","shuffle");
            binaryImageGray=rgb2gray(Lrgb);
        case 5
            I=im2gray(im1);
            BW = imbinarize(I);
            [B,L] = bwboundaries(BW,'noholes');
            binaryImage=L
            L=~L
            binaryImageGray=L
            binaryImageGray = uint8(binaryImageGray) * 255;
            

    end
% Convert the binary image to a grayscale image for display
% Create a mask for the background (non-segmented regions)
backgroundMask = ~binaryImageGray;

% Create a color version of the original image
colorImage = repmat(binaryImageGray, [1, 1, 3]);

% Set the background region to white
colorImage(backgroundMask > 0) = 255;

% Display the image with white background

cc = bwconncomp(binaryImageGray);

% Perform feature extraction for each object
stats = regionprops(cc, 'Area', 'Perimeter');
areas = zeros(1, cc.NumObjects);
    perimeters = zeros(1, cc.NumObjects);
    for k = 1:length(stats)
        areas(k) = stats(k).Area;
        perimeters(k) = stats(k).Perimeter;
    end
    
    set(handles.edit1, 'String', sprintf(mat2str(areas)));
    set(handles.edit2, 'String', sprintf(mat2str(perimeters)));

% Display the segmented image with red color
axes(handles.axes5);
imshow(colorImage);
% Display the segmented image



function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit2_Callback(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit2 as text
%        str2double(get(hObject,'String')) returns contents of edit2 as a double


% --- Executes during object creation, after setting all properties.
function edit2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
    global im1;

    % Check if the image is loaded
    if isempty(im1)
        msgbox('Please load an image first', 'Error', 'Warning');
        return
    end

    % Get the selected method from the pop-up menu
    methods = hObject.String;
    selected_method = methods{hObject.Value};

    % Convert the image to grayscale if it's not already
    if size(im1, 3) == 3
        grayImage = rgb2gray(im1);
    else
        grayImage = im1;
    end

    % Step 1: Noise reduction using Gaussian filtering
    filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

    % Step 2: Enhance contrast using histogram equalization
    enhancedImage = histeq(filteredImage);

    % Perform edge detection based on the selected method
    global a
    a=get(handles.popupmenu1,'value');
    switch a
        case 1
            edges = edge(enhancedImage, 'Canny');
        case 2
            edges = edge(enhancedImage, 'Sobel');
        case 3
            edges = edge(enhancedImage, 'Prewitt');
        case 4
            edges = edge(enhancedImage, 'Roberts');
    end

    % Display the detected edges
    axes(handles.axes3);
    imshow(edges);

% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu2.
function popupmenu2_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2
 global im1;

    % Check if the image is loaded
    if isempty(im1)
        msgbox('Please load an image first', 'Error', 'Warning');
        return
    end

    % Get the selected method from the pop-up menu
    methods = hObject.String;
    selected_method = methods{hObject.Value};

    % Convert the image to grayscale if it's not already
    if size(im1, 3) == 3
        grayImage = rgb2gray(im1);
    else
        grayImage = im1;
    end

    % Step 1: Noise reduction using Gaussian filtering
    filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

    % Step 2: Enhance contrast using histogram equalization
    enhancedImage = histeq(filteredImage);

    % Perform edge detection based on the selected method
    a=get(handles.popupmenu1,'value');
    switch a
        case 1
            edges = edge(enhancedImage, 'Canny');
        case 2
            edges = edge(enhancedImage, 'Sobel');
        case 3
            edges = edge(enhancedImage, 'Prewitt');
        case 4
            edges = edge(enhancedImage, 'Roberts');
    end

    b=get(handles.popupmenu2,'value');
    switch b
        case 1
            threshold = 0.2; % Adjust as needed
            binaryImage = edges > threshold;

% Perform morphological operations for further refinement of the segmentation
% Here, we perform closing to fill small gaps in the detected edges
            se = strel('disk', 10); % Adjust the structuring element size as needed
            binaryImage = imclose(binaryImage, se);

% Convert the binary image to a grayscale image for display
            binaryImageGray = uint8(binaryImage) * 255;
        case 2
           mask = zeros(size(grayImage));%creates a binary mask of zeroes with the same size as grayscale image
           mask(25:end-25,25:end-25) = 1;
%imshow(mask)
           bw = activecontour(edges,mask,500);

% Convert the binary image to a grayscale image for display
            binaryImageGray = bw;
        case 3
            thresholds = multithresh(edges, 4); % 'n' is the number of thresholds
            segmented_image = imquantize(grayImage, thresholds);
         

% Perform morphological operations for further refinement of the segmentation
% Here, we perform closing to fill small gaps in the detected edges
            se = strel('disk', 10); % Adjust the structuring element size as needed
            binaryImage = imclose(binaryImage, se);

% Convert the binary image to a grayscale image for display
            binaryImageGray = uint8(binaryImage) * 255;
        case 4
            T = adaptthresh(grayImage, 0.4);
            BW = imbinarize(I,T);
            
% Perform morphological operations for further refinement of the segmentation
% Here, we perform closing to fill small gaps in the detected edges
           

% Convert the binary image to a grayscale image for display
            binaryImageGray = uint8(BW) * 255;
    end
    axes(handles.axes4);
    imshow(binaryImageGray);
% --- Executes during object creation, after setting all properties.
function popupmenu2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu3.
function popupmenu3_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu3
global im1;

    % Check if the image is loaded
    if isempty(im1)
        msgbox('Please load an image first', 'Error', 'Warning');
        return
    end

    % Get the selected method from the pop-up menu
    methods = hObject.String;
    selected_method = methods{hObject.Value};

    % Convert the image to grayscale if it's not already
    if size(im1, 3) == 3
        grayImage = rgb2gray(im1);
    else
        grayImage = im1;
    end

    % Step 1: Noise reduction using Gaussian filtering
    
    global b
    b=get(handles.popupmenu3,'value');
    switch b
        case 1
            filteredImage = imgaussfilt(grayImage, 1); % Adjust the standard deviation as needed

    % Step 2: Enhance contrast using histogram equalization
            enhancedImage = histeq(filteredImage);

    % Perform edge detection based on the selected method
            global a
            a=get(handles.popupmenu1,'value');
            switch a
                case 1
                    edges = edge(enhancedImage, 'Canny');
                case 2
                    edges = edge(enhancedImage, 'Sobel');
                case 3
                    edges = edge(enhancedImage, 'Prewitt');
                case 4
                    edges = edge(enhancedImage, 'Roberts');
            end
            threshold = 0.2; % Adjust as needed
            binaryImage = edges > threshold;

% Perform morphological operations for further refinement of the segmentation
% Here, we perform closing to fill small gaps in the detected edges
            se = strel('disk', 10); % Adjust the structuring element size as needed
            binaryImage = imclose(binaryImage, se);

% Convert the binary image to a grayscale image for display
            binaryImageGray = uint8(binaryImage) * 255;
        case 2
            T = adaptthresh(im1, 0.4);
            binaryImage = imbinarize(im1,T);
            binaryImageGray = uint8(binaryImage) * 255;
        case 3
            %I = rgb2gray(im1);
%imshow(I)
%title('Original Image')
            mask = zeros(size(grayImage));%creates a binary mask of zeroes with the same size as grayscale image
            mask(25:end-25,25:end-25) = 1;
%imshow(mask)
            binaryImage = activecontour(grayImage,mask,500);
            binaryImage=~binaryImage;
            binaryImageGray = uint8(binaryImage) * 255;
        case 4
            I = im2gray(im1);
            gmag = imgradient(I);
            L = watershed(gmag);
            Lrgb = label2rgb(L);
            se = strel("disk",20);
            Io = imopen(I,se);
            Ie = imerode(I,se);
            Iobr = imreconstruct(Ie,I);
            Ioc = imclose(Io,se);
            Iobrd = imdilate(Iobr,se);
            Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
            Iobrcbr = imcomplement(Iobrcbr);
            fgm = imregionalmax(Iobrcbr);
            I2 = labeloverlay(I,fgm);
            se2 = strel(ones(5,5));
            fgm2 = imclose(fgm,se2);
            fgm3 = imerode(fgm2,se2);
            fgm4 = bwareaopen(fgm3,20);
            I3 = labeloverlay(I,fgm4);
            bw = imbinarize(Iobrcbr);
            D = bwdist(bw);
            DL = watershed(D);
            bgm = DL == 0;
            gmag2 = imimposemin(gmag, bgm | fgm4);
            L = watershed(gmag2);
            labels = imdilate(L==0,ones(3,3)) + 2*bgm + 3*fgm4;
            I4 = labeloverlay(I,labels);
            Lrgb = label2rgb(L,"jet","w","shuffle");
            binaryImageGray=Lrgb
        case 5
            I=im2gray(im1);
            BW = imbinarize(I);
            [B,L] = bwboundaries(BW,'noholes');
            L=~L
            binaryImageGray=L

    end
    axes(handles.axes4);
    imshow(binaryImageGray);

% --- Executes during object creation, after setting all properties.
function popupmenu3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
