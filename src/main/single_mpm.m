clear all;close all;clc

%% comment this section to turn warning on (but really though??!!)
w = warning('on','all');
id = w.identifier;
warning('off',id);
desired_dimension=1000;
use_pca=1;

%% Initialization
current_directory = pwd;
support_directory = [current_directory '/../../support_libraries/'];

addpath(genpath(support_directory));

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


%% Run Single-MPM

for i=1:no_class
    
    load_path=[current_directory '/../../save_folder/results/' dataset_name '/' num2str(i) '/' network '_' feature_type '.mat'];
    try
        load(load_path);
    catch ME
        if(strcmp(ME.identifier,'MATLAB:load:couldNotReadFile'))
            disp('error in naming the laod file. OR')
            disp('Run main.py with arguments of OC SVM to extract deep features of desired model')
        else
            disp(ME.message)
        end
        break
    end
    
    train_features = double(train_features);
    test_features = double(test_features);
    test_label = test_label';
    
    desired_dimension=min(1000,size(train_features,1)-1);
    
    currclass = i;
    
    if(use_pca)
[coeff,score,latent] = pca(train_features);
train_features = train_features*coeff(:,1:desired_dimension);
test_features = test_features*coeff(:,1:desired_dimension);
    end
    
    %%%% define variables
    feature_dimension=size(train_features,2);
    no_train_data=size(train_features,1);
    m_vec=mean(train_features);
    cov_mat=(train_features-repmat(m_vec, no_train_data, 1))'*(train_features-repmat(m_vec, no_train_data, 1));
    rho=0.0;
    nu=0;
    alpha=0.0001;
    cov_mat=cov_mat+eye(size(cov_mat,1))*rho;
    sig=sqrtm(cov_mat);

    cvx_begin quiet
        
        variable a(feature_dimension) 
        variable b(1)
        minimize ((norm(sig*a)))
            subject to
                m_vec*a-1 >= (sqrt(alpha/(1-alpha))+nu)*norm(sig*a);
    cvx_end

    scores = (test_features*a);
    
    eval(['save -v7.3 ' current_directory '/../../temp_files/scores.mat scores']);
    eval(['save -v7.3 ' current_directory '/../../temp_files/labels.mat test_label']);
    
    [status,cmdout] = system('python getAUC.py');
    
    results(i,1) = str2num(cmdout);
    
    disp(['Done...class ' num2str(i) ' : ' num2str(results(i))]);

end

