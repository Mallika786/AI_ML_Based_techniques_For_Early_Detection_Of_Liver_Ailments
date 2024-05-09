function varargout = AIMLGUI2(varargin)
% AIMLGUI2 MATLAB code for AIMLGUI2.fig
%      AIMLGUI2, by itself, creates a new AIMLGUI2 or raises the existing
%      singleton*.
%
%      H = AIMLGUI2 returns the handle to a new AIMLGUI2 or the handle to
%      the existing singleton*.
%
%      AIMLGUI2('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in AIMLGUI2.M with the given input arguments.
%
%      AIMLGUI2('Property','Value',...) creates a new AIMLGUI2 or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before AIMLGUI2_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to AIMLGUI2_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help AIMLGUI2

% Last Modified by GUIDE v2.5 06-Apr-2024 20:32:43

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @AIMLGUI2_OpeningFcn, ...
                   'gui_OutputFcn',  @AIMLGUI2_OutputFcn, ...
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


% --- Executes just before AIMLGUI2 is made visible.
function AIMLGUI2_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to AIMLGUI2 (see VARARGIN)

% Choose default command line output for AIMLGUI2
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes AIMLGUI2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = AIMLGUI2_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



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



function edit3_Callback(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit3 as text
%        str2double(get(hObject,'String')) returns contents of edit3 as a double


% --- Executes during object creation, after setting all properties.
function edit3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit4_Callback(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit4 as text
%        str2double(get(hObject,'String')) returns contents of edit4 as a double


% --- Executes during object creation, after setting all properties.
function edit4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit5_Callback(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit5 as text
%        str2double(get(hObject,'String')) returns contents of edit5 as a double


% --- Executes during object creation, after setting all properties.
function edit5_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit6_Callback(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit6 as text
%        str2double(get(hObject,'String')) returns contents of edit6 as a double


% --- Executes during object creation, after setting all properties.
function edit6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit7_Callback(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit7 as text
%        str2double(get(hObject,'String')) returns contents of edit7 as a double


% --- Executes during object creation, after setting all properties.
function edit7_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit8_Callback(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit8 as text
%        str2double(get(hObject,'String')) returns contents of edit8 as a double


% --- Executes during object creation, after setting all properties.
function edit8_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit8 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit9_Callback(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit9 as text
%        str2double(get(hObject,'String')) returns contents of edit9 as a double


% --- Executes during object creation, after setting all properties.
function edit9_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit9 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function edit10_Callback(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit10 as text
%        str2double(get(hObject,'String')) returns contents of edit10 as a double


% --- Executes during object creation, after setting all properties.
function edit10_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit10 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global Data;
global X_train;
global Y_train;
global X_test;
global Y_test;
% Check if Data is empty
if isempty(Data)
    % Display an error message or handle the case appropriately
    disp('Data is empty!');
    return;
end

% Check which dataset is selected
selectedDataset = get(handles.popupmenu2, 'Value'); % Assuming you have a popupmenu with tag 'popupmenu1'

% Perform preprocessing based on the selected dataset
switch selectedDataset
    case 1 % Ballooning
        % Preprocessing for SVM
        Data.Filename = grp2idx(categorical(Data.Filename));

        % Extract features (X) and target variable (Y)
        X = Data{:, 2:19};
        Y = Data{:, 20};

        % Perform ordinal encoding manually (if necessary)
        encoded_X = zeros(size(X));
        for i = 1:size(X, 2)
            [encoded_feature, ~] = grp2idx(X(:, i));
            encoded_X(:, i) = encoded_feature;
        end

        % Standardize features
        X = zscore(encoded_X);

        % Split data into training and testing sets (80% training, 20% testing)
        rng(1); % for reproducibility
        cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
        X_train = X(cv.training,:);
        Y_train = Y(cv.training,:);
        X_test = X(cv.test,:);
        Y_test = Y(cv.test,:);
        set(handles.edit1, 'String', 'Dataset has been pre-processed.');
        % Additional code for SVM can be added here

    case 4 % Naive Bayes
        % Preprocessing for Naive Bayes
        filename_labels = categorical(Data.Filename);
        filename_encoded = grp2idx(filename_labels);
        Data.Filename = filename_encoded;

        % Extract features (X) and target variable (Y)
        X = Data{:, 2:19};
        Y = Data{:, 20};

        % Perform ordinal encoding manually
        encoded_X = zeros(size(X));
        for i = 1:size(X, 2)
            [encoded_feature, ~] = grp2idx(X(:, i));
            encoded_X(:, i) = encoded_feature;
        end

        % Standardize features
        X = zscore(encoded_X);

        % Check for zero variance in predictors
        zero_var_cols = find(var(X) == 0);
        if ~isempty(zero_var_cols)
            disp('Removing predictors with zero variance...');
            X(:, zero_var_cols) = [];
            disp(['Removed columns: ', num2str(zero_var_cols)]);
        end

        % Split data into training and testing sets (80% training, 20% testing)
        rng(0); % for reproducibility
        cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
        X_train = X(cv.training,:);
        Y_train = Y(cv.training,:);
        X_test = X(cv.test,:);
        Y_test = Y(cv.test,:);
        set(handles.edit1, 'String', 'Dataset has been pre-processed.');
        % Additional code for Naive Bayes can be added here

    case 2 % Logistic Regression
        % Preprocessing for Logistic Regression
        Data.Filename = grp2idx(categorical(Data.Filename));

        % Extract features (X) and target variable (Y)
        X = Data{:, 2:19};
        Y = Data{:, 20};

        % Standardize features
        X = zscore(X);

        % Split data into training and testing sets (80% training, 20% testing)
        rng(1); % for reproducibility
        cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
        X_train = X(cv.training,:);
        Y_train = Y(cv.training,:);
        X_test = X(cv.test,:);
        Y_test = Y(cv.test,:);
        set(handles.edit1, 'String', 'Dataset has been pre-processed.');
        % Additional code for Logistic Regression can be added here

    case 3 % Decision Tree
        % Preprocessing for Decision Tree
        filename_labels = categorical(Data.Filename);
        filename_encoded = grp2idx(filename_labels);
        Data.Filename = filename_encoded;

        % Extract features (X) and target variable (Y)
        X = Data{:, 2:19};
        Y = Data{:, 20};

        % Perform ordinal encoding manually
        encoded_X = zeros(size(X));
        for i = 1:size(X, 2)
            [encoded_feature, ~] = grp2idx(X(:, i));
            encoded_X(:, i) = encoded_feature;
        end

        % Split data into training and testing sets
        rng(0); % Set random seed for reproducibility
        cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
        train_ind = cv.training;
        test_ind = cv.test;
        X_train = encoded_X(train_ind,:);
        X_test = encoded_X(test_ind,:);
        Y_train = Y(train_ind);
        Y_test = Y(test_ind);

        % Standardize features
        X_train = zscore(X_train);
        X_test = zscore(X_test);
        set(handles.edit1, 'String', 'Dataset has been pre-processed.');
        % Additional code for Decision Tree can be added here

    otherwise
        % Handle unexpected selection
        disp('Unknown dataset selected');
end



% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global Data;
global X_train;
global Y_train;
global X_test;
global Y_test;
global clf;

% Check if the training and testing data are available
if isempty(X_train) || isempty(Y_train) || isempty(X_test) || isempty(Y_test)
    % Display an error message or handle the case appropriately
    disp('Training or testing data is empty!');
    return;
end

% Initialize variables to store evaluation metrics
train_accuracy = 0;
test_accuracy = 0;
recall = 0;
specificity = 0;
error_rate = 0;
precision = 0;
f_measure = 0;
f_beta_score = 0;

% Calculate evaluation metrics based on the selected algorithm
selectedAlgorithm = get(handles.popupmenu2, 'Value');

switch selectedAlgorithm
    case 1 % SVM
        % Calculate predictions on training set
        pred_y_train = predict(clf, X_train);
        
        % Accuracy score on training set
        train_accuracy = sum(pred_y_train == Y_train) / numel(Y_train);
        
        % Predictions on testing set
        pred_y_test = predict(clf, X_test);
        
        % Accuracy score on testing set
        test_accuracy = sum(pred_y_test == Y_test) / numel(Y_test);
        
    case 4 % Naive Bayes
        % Predictions on training set
        filename_labels = categorical(Data.Filename);
        filename_encoded = grp2idx(filename_labels);
        data.Filename = filename_encoded;
        
        % Extract features (X) and target variable (Y)
        X = Data{:, 2:19};
        Y = Data{:, 20};
        
        % Perform ordinal encoding manually
        encoded_X = zeros(size(X));
        for i = 1:size(X, 2)
            [encoded_feature, ~] = grp2idx(X(:, i));
            encoded_X(:, i) = encoded_feature;
        end
        
        % Standardize features
        X = zscore(encoded_X);
        
        % Check for zero variance in predictors
        zero_var_cols = find(var(X) == 0);
        if ~isempty(zero_var_cols)
            disp('Removing predictors with zero variance...');
            X(:, zero_var_cols) = [];
            disp(['Removed columns: ', num2str(zero_var_cols)]);
        end
        
        % Split data into training and testing sets (80% training, 20% testing)
        rng(0); % for reproducibility
        cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
        X_train = X(cv.training,:);
        Y_train = Y(cv.training,:);
        X_test = X(cv.test,:);
        Y_test = Y(cv.test,:);
        
        % Train Gaussian Naive Bayes classifier
        try
            gnb = fitcnb(X_train, Y_train);
            % Evaluate classifier
            train_accuracy = sum(predict(gnb, X_train) == Y_train) / numel(Y_train);
            disp(['Training accuracy: ', num2str(train_accuracy)]);
            
            % Evaluate on testing set
            test_accuracy = sum(predict(gnb, X_test) == Y_test) / numel(Y_test);
            disp(['Testing accuracy: ', num2str(test_accuracy)]);
            pred_y_test = predict(gnb, X_test);
            pred_y_train = predict(gnb, X_train);
        catch ME
            disp(ME.message);
            disp('Unable to fit Gaussian Naive Bayes classifier due to zero variance in predictors.');
            disp('Consider preprocessing the data or using alternative approaches.');
        end  
    case 2 % Logistic Regression
        % Predictions on training set
        pred_y_train = round(predict(clf, X_train));
        
        % Predictions on testing set
        pred_y_test = round(predict(clf, X_test));
        
        % Accuracy score for training and testing sets
        train_accuracy = sum(pred_y_train == Y_train) / numel(Y_train);
        test_accuracy = sum(pred_y_test == Y_test) / numel(Y_test);
        
    case 3 % Decision Tree
        % Predictions on testing set
        pred_y_train = predict(clf, X_train);
        pred_y_test = predict(clf, X_test);
        
        % Accuracy scores
        train_accuracy = sum(predict(clf, X_train) == Y_train) / numel(Y_train);
        test_accuracy = sum(pred_y_test == Y_test) / numel(Y_test);
        
    otherwise
        % Handle unexpected selection
        disp('Unknown algorithm selected');
        return;
end

% Calculate confusion matrix
conf_mat = confusionmat(Y_test, pred_y_test);
set(handles.listbox1, 'String', {['Testing:']; num2str(conf_mat)});
conf_mat_train = confusionmat(Y_train, pred_y_train);
set(handles.listbox2, 'String', {['Training:']; num2str(conf_mat_train)});

% Calculate recall, specificity, error rate, precision, f-measure, and f-beta score
recall = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(1,2)); % True Positive Rate
specificity = conf_mat(2,2) / (conf_mat(2,1) + conf_mat(2,2)); % True Negative Rate
error_rate = (conf_mat(1,2) + conf_mat(2,1)) / sum(conf_mat(:)); % Error Rate
precision = conf_mat(1,1) / (conf_mat(1,1) + conf_mat(2,1)); % Positive Predictive Value
f_measure = (2 * precision * recall) / (precision + recall); % Harmonic Mean of Precision and Recall
beta = 0.5; % Adjust the beta parameter as needed for the F-beta score
f_beta_score = ((1 + beta^2) * precision * recall) / ((beta^2 * precision) + recall); % Weighted harmonic mean of precision and recall

% Display evaluation metrics in edittext boxes
set(handles.edit2, 'String', [num2str(train_accuracy)]);
set(handles.edit3, 'String', [num2str(test_accuracy)]);
set(handles.edit5, 'String', [num2str(recall)]);
set(handles.edit8, 'String', [num2str(specificity)]);
set(handles.edit9, 'String', [num2str(error_rate)]);
set(handles.edit4, 'String', [num2str(precision)]);
set(handles.edit7, 'String', [num2str(f_measure)]);
set(handles.edit6, 'String', [num2str(f_beta_score)]);

axes(handles.axes1); % Set current axis to axis1
plot(X_train, Y_train); % Plot X_train and Y_train
xlabel('X Train');
ylabel('Y Train');
title('Training Data');

axes(handles.axes3); % Set current axis to axis3
plot(X_test, Y_test); % Plot X_test and Y_test
xlabel('X Test');
ylabel('Y Test');
title('Testing Data');

axes(handles.axes4); % Set current axis to axis4
plot(pred_y_test, Y_test); % Plot y_pred vs y_test
xlabel('Predicted Values');
ylabel('True Values');
title('Predicted vs True Values');


% --- Executes on button press in Training.
function Training_Callback(hObject, eventdata, handles)
% hObject    handle to Training (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global Data;
global X_train;
global Y_train;
global X_test;
global Y_test;
global clf;

% Check if the training data is available
if isempty(X_train) || isempty(Y_train) || isempty(X_test) || isempty(Y_test)
    % Display an error message or handle the case appropriately
    disp('Training data is empty!');
    return;
end

% Get the selected algorithm
selectedAlgorithm = get(handles.popupmenu2, 'Value');

% Train the selected algorithm and display appropriate message in edit1
switch selectedAlgorithm
    case 1 % SVM
        try
            clf = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'BoxConstraint', 100, 'KernelScale', 1);
            set(handles.edit1, 'String', 'SVM model trained successfully.');
        catch ME
            disp(ME.message);
            disp('Unable to train SVM model.');
            set(handles.edit1, 'String', 'Error: Unable to train SVM model');
        end
    case 4 % Naive Bayes
        try
            gnb = fitcnb(X_train, Y_train);
            set(handles.edit1, 'String', 'Naive Bayes model trained successfully.');
        catch ME
            disp(ME.message);
            disp('Unable to train Gaussian Naive Bayes model.');
            set(handles.edit1, 'String', 'Error: Unable to train Naive Bayes model');
        end
    case 2 % Logistic Regression
        try
            clf = fitglm(X_train, Y_train, 'Distribution', 'binomial', 'Link', 'logit');
            set(handles.edit1, 'String', 'Logistic Regression model trained successfully.');
        catch ME
            disp(ME.message);
            disp('Unable to train Logistic Regression model.');
            set(handles.edit1, 'String', 'Error: Unable to train Logistic Regression model');
        end
    case 3 % Decision Tree
        try
            clf = fitctree(X_train, Y_train);
            set(handles.edit1, 'String', 'Decision Tree model trained successfully.');
        catch ME
            disp(ME.message);
            disp('Unable to train Decision Tree model.');
            set(handles.edit1, 'String', 'Error: Unable to train Decision Tree model');
        end
    otherwise
        % Handle unexpected selection
        disp('Unknown algorithm selected');
        set(handles.edit1, 'String', 'Error: Unknown algorithm selected');
end


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1
global Data;
selectedIndex = get(hObject, 'Value');

% Get the list of dataset options
datasetOptions = get(hObject, 'String');

% Get the selected dataset option
selectedDataset = datasetOptions{selectedIndex};

% Depending on the selected dataset, display the respective message in edit1
switch selectedDataset
    case 'Ballooning'
        % Display message in edit1
        Data = readtable('Ballooning_Data.csv');
        set(handles.edit1, 'String', 'Ballooning dataset loaded successfully.');
    case 'Fibrosis'
        % Display message in edit1
        Data = readtable('Fibrosis_Data.csv');
        set(handles.edit1, 'String', 'Fibrosis dataset loaded successfully.');
        
    case 'Inflammation'
        % Display message in edit1
        Data = readtable('Inflammation_Data.csv');
        set(handles.edit1, 'String', 'Inflammation dataset loaded successfully.');
        
    case 'Steatosis'
        % Display message in edit1
        Data = readtable('Steatosis_Data.csv');
        set(handles.edit1, 'String', 'Steatosis dataset loaded successfully.');
        
    otherwise
        % Handle unexpected selection
        set(handles.edit1, 'String', 'No dataset selected.');
end


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
selectedAlgorithm = get(handles.popupmenu2, 'Value'); % Assuming you have a popupmenu with tag 'popupmenu2'

% Display the selected algorithm in edit1
switch selectedAlgorithm
    case 1 % SVM
        set(handles.edit1, 'String', 'SVM algorithm selected.');
    case 2 % Logistic Regression
        set(handles.edit1, 'String', 'Logistic Regression algorithm selected.');
    case 3 % Decision Tree
        set(handles.edit1, 'String', 'Decision Tree algorithm selected.');
    case 4 % Naive Bayes
        set(handles.edit1, 'String', 'Naive Bayes algorithm selected.');
    otherwise
        % Handle unexpected selection
        disp('Unknown algorithm selected');
end

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu2


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


% --- Executes on selection change in listbox1.
function listbox1_Callback(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox1


% --- Executes during object creation, after setting all properties.
function listbox1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in listbox2.
function listbox2_Callback(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns listbox2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from listbox2


% --- Executes during object creation, after setting all properties.
function listbox2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to listbox2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 editHandles = findall(handles.figure1, 'Style', 'edit');
    for i = 1:numel(editHandles)
        set(editHandles(i), 'String', '');
    end

    % Clear list boxes
    listboxHandles = findall(handles.figure1, 'Style', 'listbox');
    for i = 1:numel(listboxHandles)
        set(listboxHandles(i), 'String', {});
    end

    % Clear graphs (axes)
    axesHandles = findall(handles.figure1, 'Type', 'axes');
    for i = 1:numel(axesHandles)
        cla(axesHandles(i));
    end
