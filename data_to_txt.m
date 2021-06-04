clear; clc

datafile = 'CR_Results_Set1';

load(sprintf('my_data2.mat', datafile, datafile));
label = Objective - min(Objective);
fid = fopen(sprintf('my_data.txt', datafile, datafile), 'w');
fprintf(fid, '%d\n', length(label)); %prints the number of graphs

for i = 1 : length(label)
    g = Graphs(i); %goes into each individual graph
    num_nodes = length(g.al); %gets the number of nodes
    fprintf(fid, '%d %d\n', num_nodes, label(i)); %prints the number of nodes and the graph label to the data file
    for j = 1 : num_nodes
        num_neighbors = length(g.al{j}); 
        if isempty(g.Ln2)
            fprintf(fid, '%d %d', 0, num_neighbors);
        else
            fprintf(fid, '%d %d', double(g.Ln2{j}), num_neighbors);
        end
        for k = 1 : num_neighbors
            fprintf(fid, ' %d', g.al{j}(k) - 1);
        end
        fprintf(fid, '\n');
    end
end

fclose(fid);

total = length(label);
fold_size = floor(total / 10);
p = randperm(total);
for fold = 1 : 10
    test_range = (fold - 1) * fold_size + 1 : fold * fold_size;
    train_range = [1 : (fold - 1) * fold_size, fold * fold_size + 1 : total];
    
    fid = fopen(sprintf('my_data_10fold/test_idx.txt', datafile, fold), 'w');
    for i = 1 : length(test_range)
        fprintf(fid, '%d\n', p(test_range(i)) - 1);
    end
    fclose(fid);

    fid = fopen(sprintf('my_data_10fold_idx/train_idx.txt', datafile, fold), 'w');
    for i = 1 : length(train_range)
        fprintf(fid, '%d\n', p(train_range(i)) - 1);
    end
    fclose(fid);
end