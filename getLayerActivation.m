function [act_Array] = getLayerActivation(net, images, layer)

%repmat() to produce a tile array out of the image
imgs = repmat(images, [1, 1, 1, 3]);

%Get activations for each image
act_Array = [];
for i = 1:size(images,1)
        %used imresize() to fit images into the  network input dimensions
        current_image = imresize(squeeze(imgs(i, :, :, :)), [227, 227]);
        %used activations() to get activations in response to each image
        act = activations(net, current_image, layer);
        %used reshape to produce a vector
        act = reshape(squeeze(act), 1, []);
        %Added each activation vector to the array containing activations
        %in reponse to all stimuli for each layer
        act_Array(i,:) = act;
    end
end