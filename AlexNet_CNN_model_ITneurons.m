%% PART 1 - Plot the IT Neural Response to each category

load hvm_data

%we create it_data to subselect for IT neurons
it_data = data(:, IT_idx);

%data from the first neuron (it_data(1,:)
%neural activity stored in rows
%neural index is columns

%category names (8 categories):
%Boats, Fruits, Animals, Chairs, Planes, Tables, Cars, Facesd

cat_names = string(cat_names);
obj_names = string(obj_names);

unique_cat_names = unique(cat_names);
catAnimal_indices = find(cat_names == unique_cat_names(1));
catBoat_indices = find(cat_names == unique_cat_names(2));
catCar_indices = find(cat_names == unique_cat_names(3));
catChair_indices = find(cat_names == unique_cat_names(4));
catFace_indices = find(cat_names == unique_cat_names(5));
catFruit_indices = find(cat_names == unique_cat_names(6));
catPlane_indices = find(cat_names == unique_cat_names(7));
catTable_indices = find(cat_names == unique_cat_names(8));

%Using function meanCategory get means of neural responses for each
%category
%Using function meanCategory get plots for each IT neuron response averaged
%across all stimuli
[mean1] = meanCategory(it_data,catAnimal_indices);
[mean2] = meanCategory(it_data,catBoat_indices);
[mean3] = meanCategory(it_data,catCar_indices);
[mean4] = meanCategory(it_data,catChair_indices);
[mean5] = meanCategory(it_data,catFace_indices);
[mean6] = meanCategory(it_data,catFruit_indices);
[mean7] = meanCategory(it_data,catPlane_indices);
[mean8] = meanCategory(it_data,catTable_indices);

%Plot the average IT neural response to each ccategory 
%(averaged out across images and IT neurons)
meanArray = [mean1,mean2,mean3,mean4,mean5,mean6,mean7,mean8];

figure(1);
bar([1:8],meanArray)
xlabel("(Category)","FontSize",14)
xticklabels({'Animal','Boat','Car','Chairs','Faces','Fruits','Planes','Tables'})
ylabel("Neural Response","FontSize",14)
title("Average IT Neural Response vs. Category","FontSize",14)

%% PART 2 - Report two-fold cross-validation error


%Seperate IT_data to seperate arrays based on category type

%Use a logistical regression classifier to discriminate between categories
c = 1;
cat_numbers = zeros(size(cat_names));
for i = 1: length(unique_cat_names)
obj_indices = find(cat_names == unique_cat_names(i));
cat_numbers(obj_indices) = c;
c = c+1;
end

%For categories conduct two-fold cross validation and calculate cross validation error
lda = fitcdiscr(data, cat_numbers);
cp = cvpartition(cat_numbers,'KFold', 2);
cvlda = crossval(lda,'CVPartition', cp);
ldaCVErr_categories = kfoldLoss(cvlda)

%Seperate IT_data to seperate arrays based on object type
unique_obj_names = unique(obj_names);

%Use a logistical regression classifier to discriminate between categories
o = 1;
obj_numbers = zeros(size(obj_names));
for i = 1:length(unique_obj_names)
    obj_indices = find(obj_names == unique_obj_names(i));
    obj_numbers(obj_indices) = o;
    o = o+1;
end

%For objects conduct two-fold cross validation and calculate cross validation error
lda = fitcdiscr(data, obj_numbers);
cp = cvpartition(obj_numbers,'KFold', 2);
cvlda = crossval(lda,'CVPartition', cp);
ldaCVErr_objects = kfoldLoss(cvlda)

%% PART 3 - Compute RDM matrix for IT responses (categories)

average_per_obj = [];
for i = 1:length(unique_obj_names) 
    object_idx = find(obj_names == unique_obj_names(i));
    object = it_data(object_idx,:);
    %all the neural responses for each object     
    average_per_obj(i,:) = mean(object);
    %mean of the neural responses
end

% Right order of objects
unique_obj_organized = [];
row = 1;
for i = 1:length(unique_cat_names)
    all_idx_of_cat = find(cat_numbers == i);
    obj_disorganized = obj_numbers(all_idx_of_cat);
    obj_organized = unique(sort(obj_disorganized));
    unique_obj_organized(row:row+7,1) = obj_organized;
    row = row+8;
end

% Right order of average neural responses to objects
organized_IT_responses = [];
for i = 1:length(unique_obj_organized)
    idx_IT_responses = unique_obj_organized(i);
    organized_IT_responses(i,:) = average_per_obj(idx_IT_responses,:);   
end

%transpose as corrcoef is based on columns
mat = organized_IT_responses';

%calculate dissimilarity values (1-Pearson(x,y)) for RDM
one_mat = ones(64, 64);
corr_mat = corrcoef(mat);
RDM_mat_IT = one_mat - corr_mat;

%create the RDM plot
set(figure(),'Name','Space-time RF color heat maps');
imagesc(RDM_mat_IT);
colorbar;
xticklabels({'Animal(1-8)','Boat(1-8)','Car(1-8)','Chairs(1-8)','Faces(1-8)','Fruits(1-8)','Planes(1-8)','Tables(1-8)'})
xtickangle(320)
yticklabels({'Animal(1-8)','Boat(1-8)','Car(1-8)','Chairs(1-8)','Faces(1-8)','Fruits(1-8)','Planes(1-8)','Tables(1-8)'})
title("Representational Dissimilarity Matrix (RDM) for IT responses")

axis square

%% PART 4 - Build an AlexNet convolutional neural network
%Start of part 2

load hvm_images

%Build AlexNet using layers specified in the assignment
net = alexnet();
alexnet_layer_names = ["relu3"; "relu4"; "relu5"; "fc6"; "fc7"];
alexnet_layer_num = [11 13 15 17 20];
alexnet_layer_size = [64896 64896 43264 4096 4096];

%% PART 3 - Get the actuvations for each layer

% arrayLayer1 = getLayerActivation(net,images,alexnet_layer_names(1));
% arrayLayer2 = getLayerActivation(net,images,alexnet_layer_names(2));
% arrayLayer3 = getLayerActivation(net,images,alexnet_layer_names(3));
% arrayLayer4 = getLayerActivation(net,images,alexnet_layer_names(4));
% arrayLayer5 = getLayerActivation(net,images,alexnet_layer_names(5));

%This part of the code is commented out to avoid running it each time
%(saved for later loading)
%save('layer_activations', 'arrayLayer1', 'arrayLayer2', 'arrayLayer3', 'arrayLayer4', 'arrayLayer5', '-v7.3');
%% PART 4 - Compute RDM matrix for activations

%Instead of running for an hour each time load the saved layer activations 
load layer_activations
%READ ME:
%The file layer_activations.mat was too large to send 
%PLEASE FEEL FREE TO CONTACT ME IF YOU REQUEST FILE 

% set(figure(11),'Name','Space-time RF color heat maps');
% imagesc(RDM);

%used a function to plot the RDMs for each layer activations array
%arrayLayer1,2,3,4,5
[RDM_mat1] = getLayerActRDM(unique_obj_names,obj_names,arrayLayer1,unique_cat_names,cat_numbers,obj_numbers,"Layer 1 - Relu3");
[RDM_mat2] = getLayerActRDM(unique_obj_names,obj_names,arrayLayer2,unique_cat_names,cat_numbers,obj_numbers,"Layer 2 - Relu4");
[RDM_mat3] = getLayerActRDM(unique_obj_names,obj_names,arrayLayer3,unique_cat_names,cat_numbers,obj_numbers,"Layer 3 - Relu5");
[RDM_mat4] = getLayerActRDM(unique_obj_names,obj_names,arrayLayer4,unique_cat_names,cat_numbers,obj_numbers,"Layer 4 - fc6");
[RDM_mat5] = getLayerActRDM(unique_obj_names,obj_names,arrayLayer5,unique_cat_names,cat_numbers,obj_numbers,"Layer 5 - fc7");

%% PART 5 - Compute correlation values between RDM_IT and RDM of each AlexNet Model Layer

corrLayer1 = corrcoef(RDM_mat_IT, RDM_mat1);
corrLayer2 = corrcoef(RDM_mat_IT, RDM_mat2);
corrLayer3 = corrcoef(RDM_mat_IT, RDM_mat3);
corrlayer4 = corrcoef(RDM_mat_IT, RDM_mat4);
corrLayer5 = corrcoef(RDM_mat_IT, RDM_mat5);
