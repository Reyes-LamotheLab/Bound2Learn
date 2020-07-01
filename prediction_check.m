orig_tracks = data_tracks.Segmented_Tracks;
pred_tracks = data_tracks_pred.Tracks_pred;
num_bound_class = length(pred_tracks(:,1));
ele_bound = find(Training_classification);
class_bound = orig_tracks(ele_bound,:);
pred_corr = zeros(num_bound_class,1);
for i = 1:num_bound_class
    track_num = pred_tracks(i,1);
    check_ele = find(class_bound(:,1) == track_num);
    if isempty(check_ele) ==1
        pred_corr(i,1) = 0;
    else
        pred_corr(i,1) = 1;
    end
end
per_corr = length(find(pred_corr))/num_bound_class;