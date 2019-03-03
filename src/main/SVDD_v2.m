clear all;close all;clc

%% comment this section to turn warning on
w = warning('on','all');
id = w.identifier;
warning('off',id);

%% Initialization
current_directory = pwd;
support_directory = [current_directory '/../../support_libraries'];

addpath(genpath(support_directory))

network='vggface';
dataset='um2';
feature_type='OCSVMlin_0.5';

if(dataset=='abn')
    no_class=6;
    dataset_name='abnormal';
elseif(dataset=='fd2')
    no_class=100;
    dataset_name='founder';
elseif(dataset=='um2')
    no_class=44;
    dataset_name='umdface02';
elseif(dataset=='um1')
    no_class=50;
    dataset_name='umdface01';    
elseif(dataset=='mob')
    no_class=48;
    dataset_name='mobio';
else
    disp('ERROR in dataset name.');
end

%% Run SVDD

results = zeros(no_class,1);
for i=31:no_class
    
    load_path=[current_directory '/../../save_folder/results/' dataset_name '/' num2str(i) '/' network '_' feature_type '.mat'];
    try
        load(load_path);
    catch ME
        if(strcmp(ME.identifier,'MATLAB:load:couldNotReadFile'))
            disp('error in naming the laod file. OR');
            disp(['Run main.py with arguments to produce feature of type ' feature_type '.']);
        else
            disp(ME.message);
        end
        break
    end
    
    train_features = double(train_features);
    test_features = double(test_features);
    test_label = test_label';
    
    train_features = (train_features>0).*train_features;
    test_features = (test_features>0).*test_features;
    
    model = svmtrain(ones(size(train_features,1),1), train_features,'-s 5 -t 0 -q');
    
    [~,~,scores] = svmpredict(zeros(size(test_features,1),1), test_features, model, '-q');
    
    eval(['save -v7.3 ' current_directory '/../../temp_files/scores.mat scores']);
    eval(['save -v7.3 ' current_directory '/../../temp_files/labels.mat test_label']);
    
    [status,cmdout] = system('python getAUC.py');
    
    results(i) = str2num(cmdout);
    
    disp(['Done...class ' num2str(i) ' : ' num2str(results(i))]);
    
end