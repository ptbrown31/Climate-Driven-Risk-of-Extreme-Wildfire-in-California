close all
clear all

addpath '/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/Fire/Project/matlab programs'

load('/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/Fire/Project/matlab programs/PRs_vs_Reliability_Diagram_Score_Xval_Multi_model.mat',...
    'all_vars_final_day_cmip6_altered',...
    'all_predictor_variables',...
    'temp_scenario_vars',...
    'Fire_ID',...
    'Date',...
    'Ignition_lat',...
    'Ignition_lon',...
    'Int_perim_24',...
    'binary_response',...
    'predictor_table',...
    'all_fire_probabilities_model',...
    'model_configs_to_loop_through')


load('/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/Fire/Project/matlab programs/logistic_regression_fire_probabilities.mat',...
    'all_fire_probabilities_logistic_regression',...
    'LR_labes')

perf_metric_labes = {'Log-Loss','Brier','Reliability Diagram','ROC-AUC'};
num_performance_metrics = length(perf_metric_labes);

model_type_labes = {'Neural Network','Random Forest','Logistic'};
num_types_models = length(model_type_labes);


all_fire_probabilities = NaN(length(Date),model_configs_to_loop_through,num_types_models,length(temp_scenario_vars));

all_fire_probabilities(:,:,1:2,:) = all_fire_probabilities_model;
all_fire_probabilities(:,1:5,3,:) = all_fire_probabilities_logistic_regression;

clear all_fire_probabilities_model
clear all_fire_probabilities_logistic_regression


all_fire_probabilities_2003_2020 = all_fire_probabilities(:,:,:,2);

baseline_prob = sum(binary_response)./length(binary_response);
true_labels = binary_response;

performance_scores = NaN(model_configs_to_loop_through,num_types_models,num_performance_metrics);
performance_scores_naive = NaN(model_configs_to_loop_through,num_types_models,num_performance_metrics);
performance_skill_scores = NaN(model_configs_to_loop_through,num_types_models,num_performance_metrics);

reliability_diagrams_mean_probs = {};
reliability_diagrams_test_fracs = {};

    for model_type_i = 1:num_types_models
        for model_config_i = 1:model_configs_to_loop_through

                model_prediction_probs = squeeze(all_fire_probabilities_2003_2020(:,model_config_i,model_type_i));
        
                   [log_loss_score_model,...
                       log_losses_model,...
                       log_loss_score_naive,...
                       log_losses_naive,...
                       log_loss_skill_score_model] = log_loss_skill_score(true_labels,...
                       baseline_prob,...
                       model_prediction_probs);

                       performance_scores(model_config_i,model_type_i,1) = log_loss_score_model;
                       performance_scores_naive(model_config_i,model_type_i,1) = log_loss_score_naive;
                       performance_skill_scores(model_config_i,model_type_i,1) = log_loss_skill_score_model;
        
                    brier_score  = mean((model_prediction_probs - true_labels).^2);
                    brier_score_naive = mean((baseline_prob - true_labels).^2);
                    brier_skill_score = 1 - brier_score./brier_score_naive;

                    performance_scores(model_config_i,model_type_i,2) = brier_score;
                    performance_scores_naive(model_config_i,model_type_i,2) = brier_score_naive;
                    performance_skill_scores(model_config_i,model_type_i,2) = brier_skill_score;

                    num_points_required_for_bin = 1000;
        
                    [reliability_curve_mean_bin_predictions,...
                        reliability_curve_bin_actual_fractions,...
                        reliability_curve_mean_bin_predictions_naive,...
                        reliability_score,...
                        reliability_score_naive,...
                        reliability_skill_score] = reliability_diagram_score(model_prediction_probs,...
                        true_labels,...
                        baseline_prob,...
                        num_points_required_for_bin);

                       performance_scores(model_config_i,model_type_i,3) = reliability_score;
                       performance_scores_naive(model_config_i,model_type_i,3) = reliability_score_naive;
                       performance_skill_scores(model_config_i,model_type_i,3) = reliability_skill_score;

                       reliability_diagrams_mean_probs{model_config_i,model_type_i} = reliability_curve_mean_bin_predictions;
                       reliability_diagrams_test_fracs{model_config_i,model_type_i} = reliability_curve_bin_actual_fractions;

                      posclass = 1;

                      baseline_prob_extended = NaN(length(true_labels),1);
                      baseline_prob_extended(:) = baseline_prob;
    
                      [X_model,Y_model,T_model,AUC_model,OPTROCPT_model] = perfcurve(true_labels,model_prediction_probs,posclass);
                      [X_naive,Y_naive,T_naive,AUC_naive,OPTROCPT_naive] = perfcurve(true_labels,baseline_prob_extended,posclass);
    
                      performance_scores(model_config_i,model_type_i,4) = AUC_model;
                      performance_scores_naive(model_config_i,model_type_i,4) = AUC_naive;
                      performance_skill_scores(model_config_i,model_type_i,4) = 1 - AUC_model./AUC_naive;


        end
    end

all_fire_PRs = NaN(size(all_fire_probabilities));
all_fire_FARs = NaN(size(all_fire_probabilities,1),size(all_fire_probabilities,2),size(all_fire_probabilities,3));

mean_PRs_avg_probs_first = NaN(model_configs_to_loop_through,num_types_models,length(temp_scenario_vars));

for temp_scenarios_i = 1:length(temp_scenario_vars)
    for model_type_i = 1:num_types_models
        for model_config_i = 1:model_configs_to_loop_through

            mean_PRs_avg_probs_first(model_config_i,model_type_i,temp_scenarios_i) = mean(squeeze(all_fire_probabilities(:,model_config_i,model_type_i,temp_scenarios_i)))./mean(squeeze(all_fire_probabilities(:,model_config_i,model_type_i,1)));

            all_fire_PRs(:,model_config_i,model_type_i,temp_scenarios_i) = all_fire_probabilities(:,model_config_i,model_type_i,temp_scenarios_i)./all_fire_probabilities(:,model_config_i,model_type_i,1);

            if temp_scenarios_i == 2

                all_fire_FARs(binary_response==1,model_config_i,model_type_i) = 1 - 1./all_fire_PRs(binary_response==1,model_config_i,model_type_i,temp_scenarios_i);

            end
        end
    end
end

mean_PRs = squeeze(mean(all_fire_PRs,1));
mean_FARs = squeeze(mean(all_fire_FARs,1,"omitnan")); 
mean_FARs(mean_FARs == -inf) = NaN;

min_percentiles = [33 67 90];

perc_labes = {'Top 67%',...
              '33%',...
              '10%'};

    performance_scores_all_models_pooled = NaN(model_configs_to_loop_through.*num_types_models,num_performance_metrics);
    performance_skill_scores_all_models_pooled = NaN(model_configs_to_loop_through.*num_types_models,num_performance_metrics);
    
    performance_scores_all_models_pooled(1:1000,:) = performance_scores(:,1,:);
    performance_scores_all_models_pooled(1001:2000,:) = performance_scores(:,2,:);
    performance_scores_all_models_pooled(2001:3000,:) = performance_scores(:,3,:);
    
    performance_skill_scores_all_models_pooled(1:1000,:) = performance_skill_scores(:,1,:);
    performance_skill_scores_all_models_pooled(1001:2000,:) = performance_skill_scores(:,2,:);
    performance_skill_scores_all_models_pooled(2001:3000,:) = performance_skill_scores(:,3,:);

    mean_PRs_all_models_pooled = NaN(model_configs_to_loop_through.*num_types_models,length(temp_scenario_vars));
    mean_FARs_all_models_pooled = NaN(model_configs_to_loop_through.*num_types_models,1);
    
    mean_PRs_all_models_pooled(1:1000,:) = mean_PRs_avg_probs_first(:,1,:);
    mean_PRs_all_models_pooled(1001:2000,:) = mean_PRs_avg_probs_first(:,2,:);
    mean_PRs_all_models_pooled(2001:3000,:) = mean_PRs_avg_probs_first(:,3,:);

    mean_FARs_all_models_pooled(1:1000) = mean_FARs(:,1);
    mean_FARs_all_models_pooled(1001:2000) = mean_FARs(:,2);
    mean_FARs_all_models_pooled(2001:3000) = mean_FARs(:,3);

min_percentile_perf_scores_values = NaN(length(min_percentiles),num_performance_metrics);
min_percentile_perf_skill_scores_values = NaN(length(min_percentiles),num_performance_metrics);

for perf_metric_i = 1:num_performance_metrics
    for percentile_i = 1:length(min_percentiles)
    
        min_percentile_perf_scores_values(percentile_i,perf_metric_i) = prctile(performance_scores_all_models_pooled(:,perf_metric_i),min_percentiles(percentile_i));
        min_percentile_perf_skill_scores_values(percentile_i,perf_metric_i) = prctile(performance_skill_scores_all_models_pooled(:,perf_metric_i),min_percentiles(percentile_i));
    
    end
end


percentile_mean_perf_score = NaN(length(min_percentiles),num_performance_metrics);
percentile_mean_perf_skill_score = NaN(length(min_percentiles),num_performance_metrics);

    percentile_median_PR_by_score = NaN(length(min_percentiles),num_performance_metrics,length(temp_scenario_vars));
    percentile_median_PR_by_skill_score = NaN(length(min_percentiles),num_performance_metrics,length(temp_scenario_vars));
    
    percentile_PR_lower_error_bar_by_score = NaN(length(min_percentiles),num_performance_metrics,length(temp_scenario_vars));
    percentile_PR_upper_error_bar_by_score = NaN(length(min_percentiles),num_performance_metrics,length(temp_scenario_vars));
    
    percentile_PR_lower_error_bar_by_skill_score = NaN(length(min_percentiles),num_performance_metrics,length(temp_scenario_vars));
    percentile_PR_upper_error_bar_by_skill_score = NaN(length(min_percentiles),num_performance_metrics,length(temp_scenario_vars));

    percentile_median_FAR_by_score = NaN(length(min_percentiles),num_performance_metrics);
    percentile_median_FAR_by_skill_score = NaN(length(min_percentiles));
    
    percentile_FAR_lower_error_bar_by_score = NaN(length(min_percentiles),num_performance_metrics);
    percentile_FAR_upper_error_bar_by_score = NaN(length(min_percentiles),num_performance_metrics);
    
    percentile_FAR_lower_error_bar_by_skill_score = NaN(length(min_percentiles),num_performance_metrics);
    percentile_FAR_upper_error_bar_by_skill_score = NaN(length(min_percentiles),num_performance_metrics);

min_err_bar_perc = 5;
max_err_bar_perc = 95;

for perf_metric_i = 1:num_performance_metrics
    for percentile_i = 1:length(min_percentiles)

            good_inds_scores_assoc_w_this_perc = find(performance_scores_all_models_pooled(:,perf_metric_i) >= min_percentile_perf_scores_values(percentile_i,perf_metric_i));
            good_inds_skill_scores_assoc_w_this_perc = find(performance_skill_scores_all_models_pooled(:,perf_metric_i) >= min_percentile_perf_skill_scores_values(percentile_i,perf_metric_i));
    
            percentile_mean_perf_score(percentile_i,perf_metric_i) = mean(performance_scores_all_models_pooled(good_inds_scores_assoc_w_this_perc,perf_metric_i),"omitnan");
            percentile_mean_perf_skill_score(percentile_i,perf_metric_i) = mean(performance_skill_scores_all_models_pooled(good_inds_skill_scores_assoc_w_this_perc,perf_metric_i),"omitnan");
    
        for temp_scenarios_i = 2:length(temp_scenario_vars)

            if temp_scenarios_i == 2

                percentile_median_FAR_by_score(percentile_i,perf_metric_i) = median(mean_FARs_all_models_pooled(good_inds_scores_assoc_w_this_perc),"omitnan");
        
                percentile_FAR_lower_error_bar_by_score(percentile_i,perf_metric_i) = prctile(mean_FARs_all_models_pooled(good_inds_scores_assoc_w_this_perc),min_err_bar_perc);
                percentile_FAR_upper_error_bar_by_score(percentile_i,perf_metric_i) = prctile(mean_FARs_all_models_pooled(good_inds_scores_assoc_w_this_perc),max_err_bar_perc);
    
                percentile_median_FAR_by_skill_score(percentile_i,perf_metric_i) = median(mean_FARs_all_models_pooled(good_inds_skill_scores_assoc_w_this_perc),"omitnan");
        
                percentile_FAR_lower_error_bar_by_skill_score(percentile_i,perf_metric_i) = prctile(mean_FARs_all_models_pooled(good_inds_skill_scores_assoc_w_this_perc),min_err_bar_perc);
                percentile_FAR_upper_error_bar_by_skill_score(percentile_i,perf_metric_i) = prctile(mean_FARs_all_models_pooled(good_inds_skill_scores_assoc_w_this_perc),max_err_bar_perc);   

            end

            percentile_median_PR_by_score(percentile_i,perf_metric_i,temp_scenarios_i) = median(mean_PRs_all_models_pooled(good_inds_scores_assoc_w_this_perc,temp_scenarios_i),"omitnan");
    
            percentile_PR_lower_error_bar_by_score(percentile_i,perf_metric_i,temp_scenarios_i) = prctile(mean_PRs_all_models_pooled(good_inds_scores_assoc_w_this_perc,temp_scenarios_i),min_err_bar_perc);
            percentile_PR_upper_error_bar_by_score(percentile_i,perf_metric_i,temp_scenarios_i) = prctile(mean_PRs_all_models_pooled(good_inds_scores_assoc_w_this_perc,temp_scenarios_i),max_err_bar_perc);

            percentile_median_PR_by_skill_score(percentile_i,perf_metric_i,temp_scenarios_i) = median(mean_PRs_all_models_pooled(good_inds_skill_scores_assoc_w_this_perc,temp_scenarios_i),"omitnan");
    
            percentile_PR_lower_error_bar_by_skill_score(percentile_i,perf_metric_i,temp_scenarios_i) = prctile(mean_PRs_all_models_pooled(good_inds_skill_scores_assoc_w_this_perc,temp_scenarios_i),min_err_bar_perc);
            percentile_PR_upper_error_bar_by_skill_score(percentile_i,perf_metric_i,temp_scenarios_i) = prctile(mean_PRs_all_models_pooled(good_inds_skill_scores_assoc_w_this_perc,temp_scenarios_i),max_err_bar_perc);   
        end
    end
end

    median_PR_neg_error_bars_by_score = percentile_median_PR_by_score - percentile_PR_lower_error_bar_by_score;
    median_PR_pos_error_bars_by_score = percentile_PR_upper_error_bar_by_score - percentile_median_PR_by_score;
    
    median_PR_neg_error_bars_by_skill_score = percentile_median_PR_by_skill_score - percentile_PR_lower_error_bar_by_skill_score;
    median_PR_pos_error_bars_by_skill_score = percentile_PR_upper_error_bar_by_skill_score - percentile_median_PR_by_skill_score;

    median_FAR_neg_error_bars_by_score = percentile_median_FAR_by_score - percentile_FAR_lower_error_bar_by_score;
    median_FAR_pos_error_bars_by_score = percentile_FAR_upper_error_bar_by_score - percentile_median_FAR_by_score;
    
    median_FAR_neg_error_bars_by_skill_score = percentile_median_FAR_by_skill_score - percentile_FAR_lower_error_bar_by_skill_score;
    median_FAR_pos_error_bars_by_skill_score = percentile_FAR_upper_error_bar_by_skill_score - percentile_median_FAR_by_skill_score;

perf_metric_to_use = 1;

performance_scores_skill_this_perf_metric = squeeze(performance_skill_scores(:,:,perf_metric_to_use));

top_models_i_linear = find(performance_scores_skill_this_perf_metric >= min_percentile_perf_skill_scores_values(end,perf_metric_to_use)); %this moves down columns first so the first 1000 are Neural Networks

reliability_diagrams_mean_probs_top_models = reliability_diagrams_mean_probs(top_models_i_linear);
reliability_diagrams_test_fracs_top_models = reliability_diagrams_test_fracs(top_models_i_linear);

reliability_diagrams_mean_probs_top_models_mean = mean(cell2mat(reliability_diagrams_mean_probs_top_models'),2);
reliability_diagrams_test_fracs_top_models_mean = mean(cell2mat(reliability_diagrams_test_fracs_top_models'),2);

xylim = 0.18;

    FigHandle = figure('Position', [100, 100, 600, 600]);
    set(gcf,'color',[1 1 1]);
    set(0, 'DefaultAxesFontSize',15);
    set(0,'defaultAxesFontName', 'helvetica')
    hold on

    for model_i = 1:length(top_models_i_linear)


        if top_models_i_linear(model_i) <= 1000

            plot(reliability_diagrams_mean_probs_top_models{model_i},reliability_diagrams_test_fracs_top_models{model_i},"Magenta",'LineWidth',0.5)

        end
        if top_models_i_linear(model_i) > 1000 && top_models_i_linear(model_i) <= 2000

            plot(reliability_diagrams_mean_probs_top_models{model_i},reliability_diagrams_test_fracs_top_models{model_i},"blue",'LineWidth',0.5)

        end
    end

    plot(reliability_diagrams_mean_probs_top_models_mean,reliability_diagrams_test_fracs_top_models_mean,"Black",'LineWidth',3)
    scatter(reliability_diagrams_mean_probs_top_models_mean,reliability_diagrams_test_fracs_top_models_mean,200,"Black","filled","o","MarkerEdgeColor","none")

    plot([0.01 1],[0.01 1],':r','LineWidth',4)

           h1 = plot(reliability_diagrams_mean_probs_top_models{1},reliability_diagrams_test_fracs_top_models{1},"Magenta",'LineWidth',0.5);
           h2 = plot(reliability_diagrams_mean_probs_top_models{101},reliability_diagrams_test_fracs_top_models{101},"blue",'LineWidth',0.5);
           h3 = plot(reliability_diagrams_mean_probs_top_models_mean,reliability_diagrams_test_fracs_top_models_mean,"Black",'LineWidth',3);

    legend([h1 h2 h3],{'Neural Network','Random Forest','Ensemble Mean'},"location","Northwest")

    xlim([0.01 xylim])
    ylim([0.01 xylim])

    title('Top models out-of-sample')
    xlabel('Predicted Probability of Extreme Daily Growth')
    ylabel('Observed Frequency of Extreme Daily Growth')


[top_models_i top_models_j] = find(performance_scores_skill_this_perf_metric >= min_percentile_perf_skill_scores_values(end,perf_metric_to_use));

all_fire_probabilities_top_models = [];

    for top_model_inds = 1:length(top_models_i)

        good_model_i = top_models_i(top_model_inds);
        good_model_j = top_models_j(top_model_inds);

        all_good_fire_probabilities = squeeze(all_fire_probabilities(:,good_model_i,good_model_j,:));

        all_fire_probabilities_top_models = cat(3,all_fire_probabilities_top_models,all_good_fire_probabilities);

    end

all_fire_probabilities_top_models_mean = mean(all_fire_probabilities_top_models,3);

all_fire_PRs_top_models_mean = NaN(size(all_fire_probabilities_top_models_mean));
all_fire_FARs_top_models_mean = NaN(size(all_fire_probabilities_top_models_mean,1),1);

for temp_scenarios_i = 1:length(temp_scenario_vars)

    all_fire_PRs_top_models_mean(:,temp_scenarios_i) = all_fire_probabilities_top_models_mean(:,temp_scenarios_i)./all_fire_probabilities_top_models_mean(:,1);

    if temp_scenarios_i == 2

        all_fire_FARs_top_models_mean(binary_response==1) = 1 - 1./all_fire_PRs_top_models_mean(binary_response==1,temp_scenarios_i);

    end
end

%plot

ymaxes = [2e5 0.25 5 70];
cut_off_years = 2003:3:2021;
cut_off_years_date = datetime(cut_off_years,1,1);

dot_size = 3;

large_fires_is = find(Int_perim_24 >= 10000);
temp_scenarios_i = 2;

    ymin = 0.5;

    FigHandle = figure('Position', [100, 100, 600, 900]); %[left bottom width height]
    set(gcf,'color',[1 1 1]);
    set(0, 'DefaultAxesFontSize',7);
    set(0,'defaultAxesFontName', 'helvetica')

        subplot(4,1,1)
        hold on

        for xval_i = 1:length(cut_off_years)
            plot([cut_off_years_date(xval_i) cut_off_years_date(xval_i)],[0 ymaxes(1)],'-k','LineWidth',0.5)
        end

            h1 = scatter(Date,Int_perim_24,5,"blue","filled","o","MarkerEdgeColor","none","MarkerFaceAlpha",0.5);
            h2 = scatter(Date(large_fires_is),Int_perim_24(large_fires_is),dot_size,"magenta","filled","o","MarkerEdgeColor","none","MarkerFaceAlpha",0.5);
        
            plot([Date(1) Date(end)],[10000 10000],'-k')

            legend([h1 h2],{'Daily Growth < 10,000','Daily Growth >= 10,000'},'Location','NorthWest')

        
         title('Response: Extreme Daily Growth')
         ylabel('Daily Growth (Acres)')

        subplot(4,1,2)
        hold on

        for xval_i = 1:length(cut_off_years)
            plot([cut_off_years_date(xval_i) cut_off_years_date(xval_i)],[0 ymaxes(2)],'-k','LineWidth',0.5)
        end

        [y,idx] = datasample(Date,2000);

           for idx_i = 1:length(idx)
                idx_i
                plot([Date(idx(idx_i)) Date(idx(idx_i))],[all_fire_probabilities_top_models_mean(idx(idx_i),1) all_fire_probabilities_top_models_mean(idx(idx_i),temp_scenarios_i)],'-k','LineWidth',0.1)
            end

            scatter(Date(idx),all_fire_probabilities_top_models_mean(idx,1),3,"black","filled","o","MarkerEdgeColor","none","MarkerFaceAlpha",0.6)
            scatter(Date(idx),all_fire_probabilities_top_models_mean(idx,temp_scenarios_i),3,"red","filled","o","MarkerEdgeColor","none","MarkerFaceAlpha",0.6)


         ylabel('Probability')
         title('ML Predicted Probability of Extreme Daily Growth')

        subplot(4,1,3)
        hold on

        for xval_i = 1:length(cut_off_years)
            plot([cut_off_years_date(xval_i) cut_off_years_date(xval_i)],[0 ymaxes(3)],'-k','LineWidth',0.5)
        end

            scatter(Date,all_fire_PRs_top_models_mean(:,temp_scenarios_i),2,"red","filled","o","MarkerEdgeColor","none","MarkerFaceAlpha",0.5)

            all_fire_PRs_top_models_mean_mean = mean(all_fire_PRs_top_models_mean(:,temp_scenarios_i));

            plot([Date(1) Date(end)],[1 1],':k')
        
            ylim([ymin ymaxes(3)])

         title('Probability Ratios')
         ylabel('Probability Ratio')

        subplot(4,1,4)
        hold on

        for xval_i = 1:length(cut_off_years)
            plot([cut_off_years_date(xval_i) cut_off_years_date(xval_i)],[-0.1 ymaxes(4)],'-k','LineWidth',0.5)
        end

            scatter(Date,100.*all_fire_FARs_top_models_mean,dot_size,"red","filled","o","MarkerEdgeColor","none","MarkerFaceAlpha",0.9)
        
            plot([Date(1) Date(end)],[0 0],':k')
        
            ylim([-0.1 ymaxes(4)])

         title('Fraction of Risk Attributable to Warming (%)')
         xlabel('Date')
         ylabel('Fraction of Risk Attributable to Warming (%)')

print(gcf,'-depsc','-painters','/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/Fire/Project/matlab programs/perf_score_time_series_fig')

ymins = [270 0 0 5];
ymaxs = [310 5 30 30];

ylabes = {'Temperature','Vapor Pressure Deficit','Dead Fuel Moisture, 100hr','Dead Fuel Moisture, 1000hr'};

    FigHandle = figure('Position', [100, 100, 600, 300]); %[left bottom width height]
    set(gcf,'color',[1 1 1]);
    set(0, 'DefaultAxesFontSize',9);
    set(0,'defaultAxesFontName', 'helvetica')

    predictors_to_plot = [1 2 5 6];

    for plot_i = 1:length(predictors_to_plot)

        subplot(2,2,plot_i)
        hold on

        for xval_i = 1:length(cut_off_years)
            plot([cut_off_years_date(xval_i) cut_off_years_date(xval_i)],[0 350],'-k','LineWidth',0.5)
        end

            for idx_i = 1:length(idx)
                idx_i
                plot([Date(idx(idx_i)) Date(idx(idx_i))],[all_vars_final_day_cmip6_altered(idx(idx_i),predictors_to_plot(plot_i),1) all_vars_final_day_cmip6_altered(idx(idx_i),predictors_to_plot(plot_i),temp_scenarios_i)],'-k','LineWidth',0.1)
            end

            scatter(Date(idx),all_vars_final_day_cmip6_altered(idx,predictors_to_plot(plot_i),1),3,"black","filled","o","MarkerEdgeColor","none","MarkerFaceAlpha",0.6)
           scatter(Date(idx),all_vars_final_day_cmip6_altered(idx,predictors_to_plot(plot_i),temp_scenarios_i),3,"red","filled","o","MarkerEdgeColor","none","MarkerFaceAlpha",0.6)
                
            ylim([(ymins(plot_i)) (ymaxs(plot_i))])

         if plot_i < 3; ylabel(ylabes{plot_i},"FontSize",9); end
         if plot_i == 3; ylabel({'Dead Fuel Moisture','100 hour'},"FontSize",9); end
         if plot_i == 4; ylabel({'1000 hour'},"FontSize",9); end

         if plot_i <3; set(gca,'xticklabel',[]); end

    end


top_models_expected_frequency = NaN(length(temp_scenario_vars),1);
top_models_expected_frequency_ratio = NaN(length(temp_scenario_vars),1);

how_many_extreme_growth = 200:1:950;
poiss_pdf_functions = NaN(length(how_many_extreme_growth),length(temp_scenario_vars));

for temp_scenarios_i = 1:length(temp_scenario_vars)

    top_models_expected_frequency(temp_scenarios_i) = sum(all_fire_probabilities_top_models_mean(:,temp_scenarios_i));
    top_models_expected_frequency_ratio(temp_scenarios_i) = sum(all_fire_probabilities_top_models_mean(:,temp_scenarios_i))./sum(all_fire_probabilities_top_models_mean(:,1));

    lambda = top_models_expected_frequency(temp_scenarios_i);
    poiss_pdf_functions(:,temp_scenarios_i) = poisspdf(how_many_extreme_growth,lambda);

end


most_destructive_fires_names= {'Camp',...
                                'Tubbs',...
                                'Cedar',...
                                'North',...
                                'Witch',...
                                'Woolsey',...
                                'Carr',...
                                'Glass',...
                                'LNU',...
                                'CZU',...
                                'Nuns',...
                                'Thomas',...
                                'Old',...
                                'August',...
                                'Butte'};

most_destructive_fires_lats= [39.78271086,...
                              38.54046684,...
                              32.9778678,...
                              39.85970066,...
                              33.0761229,...
                              34.20120083,...
                              40.66132274,...
                              38.57495577,...
                              38.50626438,...
                              37.22220515,...
                              38.36695407,...
                              34.38683869,...
                              34.15833425,...
                              39.72501203,...
                              38.32802841];

most_destructive_fires_lons= [-121.5323062,...
                              -122.6810056,...
                              -116.7733051,...
                              -120.9281545,...
                              -116.7283017,...
                              -118.7637489,...
                              -122.6321095,...
                              -122.49245,...
                              -122.3022317,...
                              -122.3059444,...
                              -122.3754689,...
                              -119.142131,...
                              -117.4839225,...
                              -122.726748,...
                              -120.667767];



array_of_info_for_most_destructive_fires = NaN(63,length(most_destructive_fires_names),4); 
array_of_info_for_most_destructive_fires_lifetime_mean = NaN(length(most_destructive_fires_names),4); 

array_of_days_for_most_destructive_fires = NaT(63,length(most_destructive_fires_names));

for destructive_fires_i = 1:length(most_destructive_fires_lats)

    destructive_fire_lat_now = most_destructive_fires_lats(destructive_fires_i);
    good_inds_for_this_fire = find(Ignition_lat == destructive_fire_lat_now);
    how_many_days = length(good_inds_for_this_fire);

    array_of_days_for_most_destructive_fires(1:how_many_days,destructive_fires_i) = Date(good_inds_for_this_fire);

    array_of_info_for_most_destructive_fires(1:how_many_days,destructive_fires_i,1) = Int_perim_24(good_inds_for_this_fire);
    array_of_info_for_most_destructive_fires(1:how_many_days,destructive_fires_i,2) = binary_response(good_inds_for_this_fire);
    array_of_info_for_most_destructive_fires(1:how_many_days,destructive_fires_i,3:9) = squeeze(all_fire_PRs_top_models_mean(good_inds_for_this_fire,2:end));
    array_of_info_for_most_destructive_fires(1:how_many_days,destructive_fires_i,10) = all_fire_FARs_top_models_mean(good_inds_for_this_fire);

            array_of_info_for_most_destructive_fires_lifetime_mean(destructive_fires_i,1:2) = mean(array_of_info_for_most_destructive_fires(1:how_many_days,destructive_fires_i,1:2),1);
            for temp_scenarios_i = 2:8
                new_PR_for_this_lifetime = mean(squeeze(all_fire_probabilities_top_models_mean(good_inds_for_this_fire,temp_scenarios_i)))./mean(squeeze(all_fire_probabilities_top_models_mean(good_inds_for_this_fire,1)));
                array_of_info_for_most_destructive_fires_lifetime_mean(destructive_fires_i,temp_scenarios_i+1) = new_PR_for_this_lifetime;
            end
        large_growth_new_inds = find(squeeze(array_of_info_for_most_destructive_fires(1:how_many_days,destructive_fires_i,2)) == 1);
        array_of_info_for_most_destructive_fires_lifetime_mean(destructive_fires_i,10) = mean(array_of_info_for_most_destructive_fires(large_growth_new_inds,destructive_fires_i,10),1);

end

destructive_fires_var_names = {'Growth',...
                               'Extreme Growth?',...
                               'Probability Ratio 2003-2020',...
                               'Probability Ratio 2041-2060 - SSP1-2.6',...
                               'Probability Ratio 2041-2060 - SSP2-4.5',...
                               'Probability Ratio 2041-2060 - SSP5-8.5',...
                               'Probability Ratio 2081-2100 - SSP1-2.6',...
                               'Probability Ratio 2081-2100 - SSP2-4.5',...
                               'Probability Ratio 2081-2100 - SSP5-8.5',...
                               'Fraction of Attributable Risk'};


    day_for_largest_growth_day = NaT(length(most_destructive_fires_names),1);
    
    array_of_info_for_largest_growth_day = NaN(length(most_destructive_fires_names),10);

    for destructive_fires_i = 1:length(most_destructive_fires_lats)
    
        dates_this_fire = array_of_days_for_most_destructive_fires(:,destructive_fires_i);
    
        growths_this_fire = array_of_info_for_most_destructive_fires(:,destructive_fires_i,1);
    
        largest_growth_ind = find(growths_this_fire == max(growths_this_fire));

        if ~isempty(largest_growth_ind)
    
            day_for_largest_growth_day(destructive_fires_i) = dates_this_fire(largest_growth_ind);
        
            array_of_info_for_largest_growth_day(destructive_fires_i,:) = array_of_info_for_most_destructive_fires(largest_growth_ind,destructive_fires_i,:);

        end
    end

temp_scenarios_i = 2;

    FigHandle = figure('Position', [100, 100, 400, 1200]); %[left bottom width height]
    set(gcf,'color',[1 1 1]);
    set(0, 'DefaultAxesFontSize',12);
    set(0,'defaultAxesFontName', 'helvetica')

         subaxis(2,1,1,'SpacingHoriz',0,'SpacingVert',0.1)
         hold on
         scatter(all_fire_probabilities_top_models_mean(:,1),all_fire_PRs_top_models_mean(:,temp_scenarios_i),5,...
                        squeeze(all_vars_final_day_cmip6_altered(:,10,2)),'filled','MarkerFaceAlpha',.3,'MarkerEdgeColor','none','MarkerEdgeAlpha',.3)

         plot([min(all_fire_probabilities_top_models_mean(:,1)) max(all_fire_probabilities_top_models_mean(:,1))], [1 1],'-k')
    
         set(gca, 'xScale', 'log')

         title('Preindustrial Probability vs. Probability Ratio')
         xlabel('Preindustrial Probability')
         ylabel('Probability Ratio')

         subaxis(2,1,2,'SpacingHoriz',0,'SpacingVert',0.1)

         scatter(Ignition_lon,Ignition_lat,5,...
                        squeeze(all_vars_final_day_cmip6_altered(:,10,2)),'filled','MarkerFaceAlpha',.3,'MarkerEdgeColor','none','MarkerEdgeAlpha',.3)

         title('Location of above')
         xlabel('Longitude')
         ylabel('Latitude')

temp_scenarios_i = 2;

dot_size = 5;

model_colors = {'m','b','r'};

ymaxes = [1.4 1.4 1.4 1.4];
xmins = [0.1 0.01 0.3 0.75];
xmaxs = [0.24 0.07 0.95 0.9];

text_offsets_x = [0.01 -0.055 0.01 0.01 0.01];
text_offsets_y = [-0.1 0 0.08 0 0 0];

plot_locs = [1 2 3 4];

model_markers_logistic = {'o','s','d','p','*'};

    FigHandle = figure('Position', [100, 100, 600, 300]);
    set(gcf,'color',[1 1 1]);
    set(0, 'DefaultAxesFontSize',9);
    set(0,'defaultAxesFontName', 'helvetica')
    hold on

    for perf_metric_i = 1:num_performance_metrics
    
        subaxis(2,4,plot_locs(perf_metric_i),'SpacingHoriz',0,'SpacingVert',0.02)
        hold on

        for model_type_i = 1:2

        if perf_metric_i <= 3, all_model_configs_performance_to_plot = performance_skill_scores(:,model_type_i,perf_metric_i); end
        if perf_metric_i == 4, all_model_configs_performance_to_plot = performance_scores(:,model_type_i,perf_metric_i); end

            all_model_configs_PRs = mean_PRs_avg_probs_first(:,model_type_i,temp_scenarios_i);
            
            scatter(all_model_configs_performance_to_plot,...
                    all_model_configs_PRs,...
                    dot_size,...
                    model_colors{model_type_i},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{model_type_i},'MarkerEdgeAlpha',.1)
        
        end

        model_type_i = 3;

        if perf_metric_i <= 3, all_model_configs_performance_to_plot = performance_skill_scores(:,model_type_i,perf_metric_i); end
        if perf_metric_i == 4, all_model_configs_performance_to_plot = performance_scores(:,model_type_i,perf_metric_i); end

            all_model_configs_PRs = mean_PRs_avg_probs_first(:,model_type_i,temp_scenarios_i);

            for model_config_i = 1:length(model_markers_logistic)
            
                scatter(all_model_configs_performance_to_plot(model_config_i),...
                        all_model_configs_PRs(model_config_i),...
                        dot_size,...
                        model_colors{model_type_i},'filled',model_markers_logistic{model_config_i},'MarkerFaceAlpha',.8,'MarkerEdgeColor',model_colors{model_type_i},'MarkerEdgeAlpha',.8)


            end

        
                if perf_metric_i <= 3

                    scores_to_plot = squeeze(performance_skill_scores(:,:,perf_metric_i));
                    PRs_to_plot = squeeze(mean_PRs_avg_probs_first(:,:,temp_scenarios_i));
                    
                    scatter(scores_to_plot(top_models_i_linear),...
                            PRs_to_plot(top_models_i_linear),...
                            1,...
                            'k','filled','o','MarkerFaceAlpha',1,'MarkerEdgeColor','none','MarkerEdgeAlpha',1)
                end
                if perf_metric_i == 4

                    scores_to_plot = squeeze(performance_scores(:,:,perf_metric_i));
                    PRs_to_plot = squeeze(mean_PRs_avg_probs_first(:,:,temp_scenarios_i));
                    
                    scatter(scores_to_plot(top_models_i_linear),...
                            PRs_to_plot(top_models_i_linear),...
                            1,...
                            'k','filled','o','MarkerFaceAlpha',1,'MarkerEdgeColor','none','MarkerEdgeAlpha',1)
                end
            
            

        if perf_metric_i <= 3

            errorbar(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_PR_by_skill_score(:,perf_metric_i,temp_scenarios_i),...
            median_PR_neg_error_bars_by_skill_score(:,perf_metric_i,temp_scenarios_i),...
            median_PR_pos_error_bars_by_skill_score(:,perf_metric_i,temp_scenarios_i),...
            '-k','LineWidth',1)

            plot(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_PR_by_skill_score(:,perf_metric_i,temp_scenarios_i),'-k','LineWidth',1)
            scatter(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_PR_by_skill_score(:,perf_metric_i,temp_scenarios_i),10,"black",'filled','o')

        end

        if perf_metric_i == 4

            errorbar(percentile_mean_perf_score(:,perf_metric_i),percentile_median_PR_by_score(:,perf_metric_i,temp_scenarios_i),...
            median_PR_neg_error_bars_by_score(:,perf_metric_i,temp_scenarios_i),...
            median_PR_pos_error_bars_by_score(:,perf_metric_i,temp_scenarios_i),...
            '-k','LineWidth',1)

            plot(percentile_mean_perf_score(:,perf_metric_i),percentile_median_PR_by_score(:,perf_metric_i,temp_scenarios_i),'-k','LineWidth',1)
            scatter(percentile_mean_perf_score(:,perf_metric_i),percentile_median_PR_by_score(:,perf_metric_i,temp_scenarios_i),10,"black",'filled','o')

        end

        title(perf_metric_labes{perf_metric_i})

        xlim([xmins(perf_metric_i) xmaxs(perf_metric_i)])
        ylim([1 ymaxes(perf_metric_i)])

        if perf_metric_i == 1

            ylabel({'Mean Probability Ratio'; 'Across all Fire-Days'});

        end
        if perf_metric_i > 1

            set(gca,'yticklabel',[])
        end

        if perf_metric_i == 1
            y_offs = [-0.3 -0.3 -0.25];
            x_offs = [-0.010 -0.005 -0.005];
          
            for y_offs_i = 1:length(y_offs)
 
                text(percentile_mean_perf_skill_score(y_offs_i,perf_metric_i)+x_offs(y_offs_i),1.1,perc_labes{y_offs_i},'FontSize',5)

            end


        end


        a = get(gca,'XTickLabel');  
        set(gca,'XTickLabel',a,'fontsize',7,'FontWeight','bold')
        set(gca,'XTickLabelMode','auto')

        set(gca,'xticklabel',[])

        if perf_metric_i == 1

                h1 = scatter(20,20,...
                            50,...
                            model_colors{1},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{1},'MarkerEdgeAlpha',.1);

                h2 = scatter(20,20,...
                            50,...
                            model_colors{2},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{2},'MarkerEdgeAlpha',.1);

                h3 = scatter(20,20,...
                            50,...
                            model_colors{3},'filled','MarkerFaceAlpha',.9,'MarkerEdgeColor',model_colors{3},'MarkerEdgeAlpha',.9);   

                legend([h1 h2 h3],{'Neural Network' 'Random Forest' 'Logistic'},'Location','northwest','FontSize',5,'Box','off')
        end
    end

    text_offsets_x = [0.01 -0.055 0.01 0.01 0.01];
    text_offsets_y = [-0.01 0 0.01 0 0 0];

    for perf_metric_i = 1:num_performance_metrics
  
        subaxis(2,4,plot_locs(perf_metric_i)+4,'SpacingHoriz',0,'SpacingVert',0)
        hold on

        for model_type_i = 1:2

        if perf_metric_i <= 3, all_model_configs_performance_to_plot = performance_skill_scores(:,model_type_i,perf_metric_i); end
        if perf_metric_i == 4, all_model_configs_performance_to_plot = performance_scores(:,model_type_i,perf_metric_i); end

            all_model_configs_FARs = mean_FARs(:,model_type_i);
            
            scatter(all_model_configs_performance_to_plot,...
                    all_model_configs_FARs,...
                    dot_size,...
                    model_colors{model_type_i},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{model_type_i},'MarkerEdgeAlpha',.1)
        
        end

        model_type_i = 3;

        if perf_metric_i <= 3, all_model_configs_performance_to_plot = performance_skill_scores(:,model_type_i,perf_metric_i); end
        if perf_metric_i == 4, all_model_configs_performance_to_plot = performance_scores(:,model_type_i,perf_metric_i); end

            all_model_configs_FARs = mean_FARs(:,model_type_i);

            for model_config_i = 1:length(model_markers_logistic)
            
                scatter(all_model_configs_performance_to_plot(model_config_i),...
                        all_model_configs_FARs(model_config_i),...
                        dot_size,...
                        model_colors{model_type_i},'filled',model_markers_logistic{model_config_i},'MarkerFaceAlpha',.8,'MarkerEdgeColor',model_colors{model_type_i},'MarkerEdgeAlpha',.8)
                if perf_metric_i == 1
    
                        h1 = scatter(20,20,...
                                    50,...
                                    model_colors{1},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{1},'MarkerEdgeAlpha',.1);
    
                        h2 = scatter(20,20,...
                                    50,...
                                    model_colors{2},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{2},'MarkerEdgeAlpha',.1);

                        h3 = scatter(20,20,...
                                    50,...
                                    model_colors{3},'filled','MarkerFaceAlpha',.9,'MarkerEdgeColor',model_colors{3},'MarkerEdgeAlpha',.9);   

                end
            end

                if perf_metric_i <= 3

                    scores_to_plot = squeeze(performance_skill_scores(:,:,perf_metric_i));
                    FARs_to_plot = squeeze(mean_FARs(:,:));
                    
                    scatter(scores_to_plot(top_models_i_linear),...
                            FARs_to_plot(top_models_i_linear),...
                            1,...
                            'k','filled','o','MarkerFaceAlpha',1,'MarkerEdgeColor','none','MarkerEdgeAlpha',1)
                end
                if perf_metric_i == 4

                    scores_to_plot = squeeze(performance_scores(:,:,perf_metric_i));
                    FARs_to_plot = squeeze(mean_FARs(:,:));
                    
                    scatter(scores_to_plot(top_models_i_linear),...
                            FARs_to_plot(top_models_i_linear),...
                            1,...
                            'k','filled','o','MarkerFaceAlpha',1,'MarkerEdgeColor','none','MarkerEdgeAlpha',1)
                end

        if perf_metric_i <= 3

            errorbar(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_FAR_by_skill_score(:,perf_metric_i),...
            median_FAR_neg_error_bars_by_skill_score(:,perf_metric_i),...
            median_FAR_pos_error_bars_by_skill_score(:,perf_metric_i),...
            '-k','LineWidth',1)

            plot(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_FAR_by_skill_score(:,perf_metric_i),'-k','LineWidth',1)
            scatter(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_FAR_by_skill_score(:,perf_metric_i),10,"black",'filled','o')

        end

        if perf_metric_i == 4

            errorbar(percentile_mean_perf_score(:,perf_metric_i),percentile_median_FAR_by_score(:,perf_metric_i),...
            median_FAR_neg_error_bars_by_score(:,perf_metric_i),...
            median_FAR_pos_error_bars_by_score(:,perf_metric_i),...
            '-k','LineWidth',1)

            plot(percentile_mean_perf_score(:,perf_metric_i),percentile_median_FAR_by_score(:,perf_metric_i),'-k','LineWidth',1)
            scatter(percentile_mean_perf_score(:,perf_metric_i),percentile_median_FAR_by_score(:,perf_metric_i),10,"black",'filled','o')

        end

        xlim([xmins(perf_metric_i) xmaxs(perf_metric_i)])
        ylim([0 0.3])

        if perf_metric_i > 1

            set(gca,'yticklabel',[])
        end      

        if perf_metric_i == 1; ylabel({'Mean Fraction of Attributable Risk','Across all Extreme Growth Fire-Days'},'FontSize',7); end

        if perf_metric_i <= 3; xlabel('Skill Score','FontSize',7); end
        if perf_metric_i == 4; xlabel('Score','FontSize',7); end

        a = get(gca,'XTickLabel');  
        set(gca,'XTickLabel',a,'fontsize',7,'FontWeight','bold')
        set(gca,'XTickLabelMode','auto')

    end

temp_scens_to_plot = [3 6 5 8];

temp_scenarios_i = 2;

dot_size = 1;

model_colors = {'m','b','r'};

ymaxes = [2.1 2.1 2.1 2.1];
xmins = [0.1 0.01 0.3 0.75];
xmaxs = [0.27 0.07 0.9 0.95];

text_offsets_x = [0.01 -0.055 0.01 0.01 0.01];
text_offsets_y = [-0.1 0 0.08 0 0 0];

plot_locs = [1 2 3 4];

model_markers_logistic = {'o','s','d','p','*'};

    plot_count = 1;

    FigHandle = figure('Position', [100, 100, 600, 800]); %[left bottom width height]
    set(gcf,'color',[1 1 1]);
    set(0, 'DefaultAxesFontSize',7);
    set(0,'defaultAxesFontName', 'helvetica')
    hold on

for sub_temp_scen_i = 1:length(temp_scens_to_plot)

    temp_scenarios_i = temp_scens_to_plot(sub_temp_scen_i);

    for perf_metric_i = 1:num_performance_metrics
    
        subplot(4,4,plot_count)
        hold on

        for model_type_i = 1:2

        if perf_metric_i <= 3, all_model_configs_performance_to_plot = performance_skill_scores(:,model_type_i,perf_metric_i); end
        if perf_metric_i == 4, all_model_configs_performance_to_plot = performance_scores(:,model_type_i,perf_metric_i); end

            all_model_configs_PRs = mean_PRs_avg_probs_first(:,model_type_i,temp_scenarios_i);
            
            scatter(all_model_configs_performance_to_plot,...
                    all_model_configs_PRs,...
                    dot_size,...
                    model_colors{model_type_i},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{model_type_i},'MarkerEdgeAlpha',.1)
        
        end

        model_type_i = 3;

        if perf_metric_i <= 3, all_model_configs_performance_to_plot = performance_skill_scores(:,model_type_i,perf_metric_i); end
        if perf_metric_i == 4, all_model_configs_performance_to_plot = performance_scores(:,model_type_i,perf_metric_i); end

            all_model_configs_PRs = mean_PRs_avg_probs_first(:,model_type_i,temp_scenarios_i);

            for model_config_i = 1:length(model_markers_logistic)
            
                scatter(all_model_configs_performance_to_plot(model_config_i),...
                        all_model_configs_PRs(model_config_i),...
                        dot_size,...
                        model_colors{model_type_i},'filled',model_markers_logistic{model_config_i},'MarkerFaceAlpha',.8,'MarkerEdgeColor',model_colors{model_type_i},'MarkerEdgeAlpha',.8)

                if perf_metric_i == 1
    
                        h1 = scatter(20,20,...
                                    50,...
                                    model_colors{1},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{1},'MarkerEdgeAlpha',.1);
    
                        h2 = scatter(20,20,...
                                    50,...
                                    model_colors{2},'filled','MarkerFaceAlpha',.1,'MarkerEdgeColor',model_colors{2},'MarkerEdgeAlpha',.1);

                        h3 = scatter(20,20,...
                                    50,...
                                    model_colors{3},'filled','MarkerFaceAlpha',.9,'MarkerEdgeColor',model_colors{3},'MarkerEdgeAlpha',.9);   

                end
            end

                        if perf_metric_i <= 3

                    scores_to_plot = squeeze(performance_skill_scores(:,:,perf_metric_i));
                    PRs_to_plot = squeeze(mean_PRs_avg_probs_first(:,:,temp_scenarios_i));
                    
                    scatter(scores_to_plot(top_models_i_linear),...
                            PRs_to_plot(top_models_i_linear),...
                            1,...
                            'k','filled','o','MarkerFaceAlpha',1,'MarkerEdgeColor','none','MarkerEdgeAlpha',1)
                end
                if perf_metric_i == 4

                    scores_to_plot = squeeze(performance_scores(:,:,perf_metric_i));
                    PRs_to_plot = squeeze(mean_PRs_avg_probs_first(:,:,temp_scenarios_i));
                    
                    scatter(scores_to_plot(top_models_i_linear),...
                            PRs_to_plot(top_models_i_linear),...
                            1,...
                            'k','filled','o','MarkerFaceAlpha',1,'MarkerEdgeColor','none','MarkerEdgeAlpha',1)
                end
            
            
        if perf_metric_i <= 3

            errorbar(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_PR_by_skill_score(:,perf_metric_i,temp_scenarios_i),...
            median_PR_neg_error_bars_by_skill_score(:,perf_metric_i,temp_scenarios_i),...
            median_PR_pos_error_bars_by_skill_score(:,perf_metric_i,temp_scenarios_i),...
            '-k','LineWidth',1)

            plot(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_PR_by_skill_score(:,perf_metric_i,temp_scenarios_i),'-k','LineWidth',1)
            scatter(percentile_mean_perf_skill_score(:,perf_metric_i),percentile_median_PR_by_skill_score(:,perf_metric_i,temp_scenarios_i),10,"black",'filled','o')

        end

        if perf_metric_i == 4

            errorbar(percentile_mean_perf_score(:,perf_metric_i),percentile_median_PR_by_score(:,perf_metric_i,temp_scenarios_i),...
            median_PR_neg_error_bars_by_score(:,perf_metric_i,temp_scenarios_i),...
            median_PR_pos_error_bars_by_score(:,perf_metric_i,temp_scenarios_i),...
            '-k','LineWidth',1)

            plot(percentile_mean_perf_score(:,perf_metric_i),percentile_median_PR_by_score(:,perf_metric_i,temp_scenarios_i),'-k','LineWidth',1)
            scatter(percentile_mean_perf_score(:,perf_metric_i),percentile_median_PR_by_score(:,perf_metric_i,temp_scenarios_i),10,"black",'filled','o')

        end

        if plot_count <= 4; title(perf_metric_labes{perf_metric_i}); end

        xlim([xmins(perf_metric_i) xmaxs(perf_metric_i)])
        ylim([1 ymaxes(perf_metric_i)])

        if perf_metric_i == 1

            ylabel('Probability Ratio');

        end
        if perf_metric_i > 1

            set(gca,'yticklabel',[])
        end

        a = get(gca,'XTickLabel');  
        set(gca,'XTickLabel',a,'fontsize',7,'FontWeight','bold')
        set(gca,'XTickLabelMode','auto')

        if plot_count <= 12; set(gca,'xticklabel',[]); end
        if plot_count > 12; xlabel('Score'); end

        if plot_count >= 12; ymaxes = [5 5 5 5]; end

        plot_count = plot_count + 1;

    end
end

day_to_read_i = 5;

years_of_interest = [2020 2017 2003 2007 2008];
month_of_interest = [8 10 10 10 06];
days_of_interest = [20 9 26 22 22];

addpath '/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/wind_solar_droughts/clim_expl_data/'

load(strcat('/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/Fire/Project/matlab programs/prob_large_growth_temp_CMIP6_gridded_day_of_interest_',...
            num2str(years_of_interest(day_to_read_i)),'_',...
            num2str(month_of_interest(day_to_read_i)),'_',...
            num2str(days_of_interest(day_to_read_i)),'_',...
            '.mat'),...
            'all_vars_final_day_cmip6_gridded_altered',...
            'all_predictor_variables',...
            'temp_scenario_vars')

all_predictor_variables_figs = ["temperature (K)",...
                           "VPD (hPa)",...
                           "mean precip (mm/hr)",...
                           "wind speed (m/s)",...
                           "100 hour DFM (%)",...
                           "1000 hour DFM (%)",...
                           "slope (m/2km)",...
                           "aspect (category)",...
                           "elevation (m)",...
                           "land use (category)",...
                           "vegetation frac (%)"];

load('/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/Fire/Project/matlab programs/read_org_background_temp_CMIP6.mat',...
'wrf_lat_2d',...
'wrf_lon_2d')

wrf_lat_2d = double(wrf_lat_2d);
wrf_lon_2d = double(wrf_lon_2d);

background = squeeze(all_vars_final_day_cmip6_gridded_altered(:,:,11,2));

good_destructive_inds_reordered = [7 14 1 4 2 8 9 15 10 12 6];

text_lats = linspace(41.95,38,11);
text_lons = zeros(1,11)-119;

temp_scenarios_i = 2;

        figure('Position', [100, 100, 600, 800]); %[left bottom width height]
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',14);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        subaxis(2,1,1,'SpacingHoriz',0,'SpacingVert',0)
        hold on

                axesm('robinson',...
                'Frame', 'on',...
                'Grid', 'off',...
                'maplatlim',[min(min(wrf_lat_2d)) max(max(wrf_lat_2d))],...
                'maplonlim',[min(min(wrf_lon_2d)) max(max(wrf_lon_2d))])
                tightmap
        
                pcolorm(wrf_lat_2d,wrf_lon_2d,background,'LineStyle','none');        
                caxis([0 250])
                 
                  min_val = 0.9;
                  inc = 0.1;
                  max_val = 4;
                  threshs = min_val:inc:max_val;

                  [y,idx2] = datasample(Date,3000);
                  
                  for threshs_i = 1:length(threshs)
                      PR_inds_in_this_range = find(squeeze(all_fire_PRs_top_models_mean(idx2,temp_scenarios_i)) > (threshs(threshs_i)-inc) & squeeze(all_fire_PRs_top_models_mean(idx2,temp_scenarios_i)) <= threshs(threshs_i));
                      scatterm(Ignition_lat(PR_inds_in_this_range),Ignition_lon(PR_inds_in_this_range),7,'o','filled','MarkerFaceColor',[(threshs(threshs_i)/max_val) 0 0],'MarkerFaceAlpha',0.9,"markeredgecolor",'none');
                  end


                  for dest_i = 1:length(good_destructive_inds_reordered)

                      textm(text_lats(dest_i),...
                          text_lons(dest_i),...
                          strcat(most_destructive_fires_names{good_destructive_inds_reordered(dest_i)},', ',...
                          num2str(year(day_for_largest_growth_day(good_destructive_inds_reordered(dest_i)))),', ',...
                          num2str(round(array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),3),2))),...
                          'FontSize',9,"FontWeight","bold")

                          plotm([text_lats(dest_i) most_destructive_fires_lats(good_destructive_inds_reordered(dest_i))],...
                                [text_lons(dest_i) most_destructive_fires_lons(good_destructive_inds_reordered(dest_i))],'-k')
                  end

                  ylabel('Probability Ratio (Historical vs. Preindustrial)')
                  title('Historical')

                  colormap(brewermap([],'greys'))

        subaxis(2,1,2,'SpacingHoriz',0,'SpacingVert',0)
        hold on

                axesm('robinson',...
                'Frame', 'on',...
                'Grid', 'off',...
                'maplatlim',[min(min(wrf_lat_2d)) max(max(wrf_lat_2d))],...
                'maplonlim',[min(min(wrf_lon_2d)) max(max(wrf_lon_2d))])
                tightmap
        
                pcolorm(wrf_lat_2d,wrf_lon_2d,background,'LineStyle','none');        
                caxis([0 250])

                  min_val = 0;
                  inc = 0.05;
                  max_val = 0.7;
                  threshs = min_val:inc:max_val;
                  
                  for threshs_i = 1:length(threshs)
                      far_inds_in_this_range = find(all_fire_FARs_top_models_mean > (threshs(threshs_i)-inc) & all_fire_FARs_top_models_mean <= threshs(threshs_i));
                      scatterm(Ignition_lat(far_inds_in_this_range),Ignition_lon(far_inds_in_this_range),20,'o','filled','MarkerFaceColor',[(threshs(threshs_i)/max_val) 0 0],'MarkerFaceAlpha',0.9);
                  end

                  for dest_i = 1:length(good_destructive_inds_reordered)

                      textm(text_lats(dest_i),...
                          text_lons(dest_i),...
                          strcat(most_destructive_fires_names{good_destructive_inds_reordered(dest_i)},', ',...
                          num2str(year(day_for_largest_growth_day(good_destructive_inds_reordered(dest_i)))),', ',...
                          num2str(round(100.*array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),10))),...
                          '%'),...
                          'FontSize',9,"FontWeight","bold")

                          plotm([text_lats(dest_i) most_destructive_fires_lats(good_destructive_inds_reordered(dest_i))],...
                                [text_lons(dest_i) most_destructive_fires_lons(good_destructive_inds_reordered(dest_i))],'-k')
                  end

                  ylabel('Fraction of Risk Attributable to Warming (%)')

                  colormap(brewermap([],'greys'))

        figure('Position', [100, 100, 560, 950]); %[left bottom width height]
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',11);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        temp_scens_to_plot = [3 6 5 8];
        locs_to_plot = [1 2 3 4];

        for temp_scens_to_plot_i = 1:length(temp_scens_to_plot)

            subaxis(3,2,locs_to_plot(temp_scens_to_plot_i),'SpacingHoriz',0,'SpacingVert',0)

                    axesm('robinson',...
                    'Frame', 'off',...
                    'Grid', 'off',...
                    'maplatlim',[min(min(wrf_lat_2d)) max(max(wrf_lat_2d))],...
                    'maplonlim',[min(min(wrf_lon_2d)) max(max(wrf_lon_2d))])
                    tightmap
            
                    pcolorm(wrf_lat_2d,wrf_lon_2d,background,'LineStyle','none');        
                    caxis([0 200])

                      min_val = 0.9;
                      inc = 0.1;
                      max_val = 10;
                      threshs = min_val:inc:max_val;
                      
                      for threshs_i = 1:length(threshs)
                          PR_inds_in_this_range = find(squeeze(all_fire_PRs_top_models_mean(idx2,temp_scens_to_plot(temp_scens_to_plot_i))) > (threshs(threshs_i)-inc) & squeeze(all_fire_PRs_top_models_mean(idx2,temp_scens_to_plot(temp_scens_to_plot_i))) <= threshs(threshs_i));
                          scatterm(Ignition_lat(PR_inds_in_this_range),Ignition_lon(PR_inds_in_this_range),6,'o','filled','MarkerFaceColor',[(threshs(threshs_i)/max_val) 0 0],'MarkerFaceAlpha',0.8);
                      end

                         for dest_i = 1:length(good_destructive_inds_reordered)
    
                              textm(text_lats(dest_i),...
                                  text_lons(dest_i),...
                                  strcat(most_destructive_fires_names{good_destructive_inds_reordered(dest_i)},', ',...
                                  num2str(year(day_for_largest_growth_day(good_destructive_inds_reordered(dest_i)))),', ',...
                                  num2str(round(array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),temp_scens_to_plot(temp_scens_to_plot_i)+1),2))),...
                                  'FontSize',6.5,"FontWeight","bold")

                                  plotm([text_lats(dest_i) most_destructive_fires_lats(good_destructive_inds_reordered(dest_i))],...
                                        [text_lons(dest_i) most_destructive_fires_lons(good_destructive_inds_reordered(dest_i))],'-k')
                         end

                  colormap(brewermap([],'greys'))

                      if locs_to_plot(temp_scens_to_plot_i) == 1; ylabel('Low Emissions'); end
                      if locs_to_plot(temp_scens_to_plot_i) == 3; ylabel('Very High Emissions'); end
                      if locs_to_plot(temp_scens_to_plot_i) == 1; title('Mid-Century'); end
                      if locs_to_plot(temp_scens_to_plot_i) == 2; title('End-Century'); end
        end

pois_maxs = max(poiss_pdf_functions,[],1);

yoffset = 0.0015;
xoffset = -20;

yminPlot = 0;
ymaxPlot = 0.033;
xminPlot = 200;
xmaxPlot = 1100;

    subaxis(3,1,3,'SpacingHoriz',0.3,'SpacingVert',0.1)
    hold on

    plot([sum(binary_response) sum(binary_response)],[yminPlot ymaxPlot],':k','LineWidth',1)

    h1 = plot(how_many_extreme_growth,poiss_pdf_functions(:,1),'-b','LineWidth',2);
    h2 = plot(how_many_extreme_growth,poiss_pdf_functions(:,2),'-k','LineWidth',2);

    h3 = plot(how_many_extreme_growth,poiss_pdf_functions(:,3),'color',[0.9290 0.6940 0.1250],'LineWidth',2,LineStyle='-.');
    h4 = plot(how_many_extreme_growth,poiss_pdf_functions(:,5),'color',[0.6350 0.0780 0.1840],'LineWidth',2,LineStyle='-.');
 
    h5 = plot(how_many_extreme_growth,poiss_pdf_functions(:,6),'color',[0.9290 0.6940 0.1250],'LineWidth',2,LineStyle='-');
    h6 = plot(how_many_extreme_growth,poiss_pdf_functions(:,8),'color',[0.6350 0.0780 0.1840],'LineWidth',2,LineStyle='-');

    legend([h1 h2 h3 h5 h4 h6],{'Preindustrial',...
        'Historical, Predicted',...
        'Low Emissions, Mid-Century',...
        'Low Emissions, End-Century',...
        'Very High Emissions, Mid-Century',...
        'Very High Emissions, End-Century'},...
        'Location','Best','Box','off','Fontsize',9)

        text(top_models_expected_frequency(1)+xoffset,pois_maxs(1)+yoffset,num2str(round(top_models_expected_frequency(1))),"FontSize",12,"Color","blue")
        text(top_models_expected_frequency(2)+xoffset-5,pois_maxs(2)+yoffset,num2str(round(top_models_expected_frequency(2))),"FontSize",12,"Color","black")
        text(top_models_expected_frequency(3)+xoffset-25,pois_maxs(3)+yoffset,num2str(round(top_models_expected_frequency(3))),"FontSize",12,"Color",[0.9290 0.6940 0.1250])
        text(top_models_expected_frequency(5)+xoffset,pois_maxs(5)+yoffset,num2str(round(top_models_expected_frequency(5))),"FontSize",12,"Color",[0.6350 0.0780 0.1840])
        text(top_models_expected_frequency(6)+xoffset+20,pois_maxs(6)+yoffset,num2str(round(top_models_expected_frequency(6))),"FontSize",12,"Color",[0.9290 0.6940 0.1250])
        text(top_models_expected_frequency(8)+xoffset,pois_maxs(8)+yoffset,num2str(round(top_models_expected_frequency(8))),"FontSize",12,"Color",[0.6350 0.0780 0.1840])


    xlabel('Total Number of Extreme Daily Growth Occurrences')
    ylabel('Probability Density')
    title('Predicted Frequency of Extreme Growth Days')

    ylim([yminPlot ymaxPlot])
    xlim([xminPlot xmaxPlot])
        
        figure('Position', [100, 100, 700, 500]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',10);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on
            PRs_bins = 0.8:0.2:10;
            
            subaxis(2,1,1,'SpacingHoriz',0.1,'SpacingVert',0.05)
            hold on
        
            histogram(all_fire_PRs_top_models_mean(:,2),PRs_bins,...
                        'Normalization',...
                        'probability',...
                        'FaceColor',...
                        'k',...
                        'EdgeColor',...
                        'none',...
                        'EdgeAlpha',...
                        0.5,...
                        'FaceAlpha',...
                        0.5);
            histogram(all_fire_PRs_top_models_mean(:,3),PRs_bins,...
                        'Normalization',...
                        'probability',...
                        'FaceColor',...
                        'b',...
                        'EdgeColor',...
                        'none',...
                        'EdgeAlpha',...
                        0.5,...
                        'FaceAlpha',...
                        0.5);
            histogram(all_fire_PRs_top_models_mean(:,6),PRs_bins,...
                        'Normalization',...
                        'probability',...
                        'FaceColor',...
                        'r',...
                        'EdgeColor',...
                        'none',...
                        'EdgeAlpha',...
                        0.5,...
                        'FaceAlpha',...
                        0.5);
        
            plot([mean(all_fire_PRs_top_models_mean(:,2)) mean(all_fire_PRs_top_models_mean(:,2))],[0 0.45],'-k','LineWidth',1)
            plot([mean(all_fire_PRs_top_models_mean(:,3)) mean(all_fire_PRs_top_models_mean(:,3))],[0 0.45],'-b','LineWidth',1)
            plot([mean(all_fire_PRs_top_models_mean(:,6)) mean(all_fire_PRs_top_models_mean(:,6))],[0 0.45],'-r','LineWidth',1)
            plot([1 1],[0 0.45],':k','LineWidth',0.5)
       
            ylabel('Fraction')
        
            legend({'Historical','Mid-Century','End-Century'})

            set(gca,'xticklabel',[])
                
            subaxis(2,1,2,'SpacingHoriz',0.1,'SpacingVert',0)
            hold on
            histogram(all_fire_PRs_top_models_mean(:,2),PRs_bins,...
                        'Normalization',...
                        'probability',...
                        'FaceColor',...
                        'k',...
                        'EdgeColor',...
                        'none',...
                        'EdgeAlpha',...
                        0.5,...
                        'FaceAlpha',...
                        0.5);
            histogram(all_fire_PRs_top_models_mean(:,5),PRs_bins,...
                        'Normalization',...
                        'probability',...
                        'FaceColor',...
                        'b',...
                        'EdgeColor',...
                        'none',...
                        'EdgeAlpha',...
                        0.5,...
                        'FaceAlpha',...
                        0.5);
            histogram(all_fire_PRs_top_models_mean(:,8),PRs_bins,...
                        'Normalization',...
                        'probability',...
                        'FaceColor',...
                        'r',...
                        'EdgeColor',...
                        'none',...
                        'EdgeAlpha',...
                        0.5,...
                        'FaceAlpha',...
                        0.5);
        
            plot([mean(all_fire_PRs_top_models_mean(:,2)) mean(all_fire_PRs_top_models_mean(:,2))],[0 0.45],'-k','LineWidth',1)
            plot([mean(all_fire_PRs_top_models_mean(:,5)) mean(all_fire_PRs_top_models_mean(:,5))],[0 0.45],'-b','LineWidth',1)
            plot([mean(all_fire_PRs_top_models_mean(:,8)) mean(all_fire_PRs_top_models_mean(:,8))],[0 0.45],'-r','LineWidth',1)
            plot([1 1],[0 0.45],':k','LineWidth',3)
        
                ylabel('Fraction')
                xlabel('Probability Ratio Distribution','Fontsize',8)

        figure('Position', [100, 100, 600, 600]); %[left bottom width height]
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',15);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

            PRs_bins = 0:0.1:4;
            
                subaxis(2,2,1,'SpacingHoriz',0.1,'SpacingVert',0.1)
                hold on
    
                pd = fitdist(squeeze(all_fire_PRs_top_models_mean(:,2)),'Kernel','Kernel','normal');
                kernal_y = pdf(pd,PRs_bins);
                plot(PRs_bins,kernal_y./(sum(kernal_y)),'LineWidth',4,"color","black");

                mean_to_plot = mean(all_fire_PRs_top_models_mean(:,2));
                text(mean_to_plot+0.1,0.18,num2str(round(mean_to_plot,2)),"fontsize",20)

                plot([mean_to_plot mean_to_plot],[0 0.25],"color","Black",'LineWidth',4,'linestyle',":")
                plot([1 1],[0 0.25],'--k','LineWidth',0.5)
           
                ylabel('Relative Frequency')
                xlabel('Probability Ratio')

            PRs_bins = -0.2:0.05:1;
            
                subaxis(2,2,2,'SpacingHoriz',0.1,'SpacingVert',0.1)
                hold on
    
                pd = fitdist(squeeze(100.*all_fire_FARs_top_models_mean),'Kernel','Kernel','normal');
                kernal_y = pdf(pd,100.*PRs_bins);
                plot(100.*PRs_bins,kernal_y./(sum(kernal_y)),'LineWidth',4,"color","black");

                mean_to_plot = mean(100.*all_fire_FARs_top_models_mean,"omitnan");
                text(mean_to_plot+5,0.18,strcat(num2str(round(mean_to_plot)),"%"),"fontsize",20)
                plot([mean_to_plot mean_to_plot],[0 0.25],"color","Black",'LineWidth',4,'linestyle',":")
                
                xlim([-15 100])
                
                plot([0 0],[0 0.25],'--k','LineWidth',0.5)

                ylabel('Relative Frequency')
                xlabel('Fraction of Risk Attributable to Warming (%)')

inds_2_plot = [3 6 5 8];

        figure('Position', [100, 100, 600, 600]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',15);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

            PRs_bins = 0:0.1:10;
            
            for plot_i = 1:length(inds_2_plot)

                subaxis(2,2,plot_i,'SpacingHoriz',0.1,'SpacingVert',0.1)
                hold on
    
                pd = fitdist(squeeze(all_fire_PRs_top_models_mean(:,2)),'Kernel','Kernel','normal');
                kernal_y = pdf(pd,PRs_bins);
                plot(PRs_bins,kernal_y./(sum(kernal_y)),'LineWidth',2,"color","black");
    
                pd = fitdist(squeeze(all_fire_PRs_top_models_mean(:,inds_2_plot(plot_i))),'Kernel','Kernel','epanechnikov');
                kernal_y = pdf(pd,PRs_bins);
    
                if inds_2_plot(plot_i) == 3 || inds_2_plot(plot_i) == 6; plot(PRs_bins,kernal_y./(sum(kernal_y)),'LineWidth',4,"color",[0.9290 0.6940 0.1250]); end
                if inds_2_plot(plot_i) == 5 || inds_2_plot(plot_i) == 8; plot(PRs_bins,kernal_y./(sum(kernal_y)),'LineWidth',4,"color",[0.6350 0.0780 0.1840]); end
            
                plot([mean(all_fire_PRs_top_models_mean(:,2)) mean(all_fire_PRs_top_models_mean(:,2))],[0 0.25],"color","Black",'LineWidth',2,'linestyle',":")
                if inds_2_plot(plot_i) == 3 || inds_2_plot(plot_i) == 6 
                    mean_to_plot = mean(all_fire_PRs_top_models_mean(:,inds_2_plot(plot_i)));
                    plot([mean_to_plot mean_to_plot],[0 0.25],"color",[0.9290 0.6940 0.1250],'LineWidth',4,'linestyle',":"); 
                    text(mean_to_plot+0.3,0.1,num2str(round(mean_to_plot,2)),"color",[0.9290 0.6940 0.1250],"fontsize",20)
                end
                if inds_2_plot(plot_i) == 5 || inds_2_plot(plot_i) == 8
                    mean_to_plot = mean(all_fire_PRs_top_models_mean(:,inds_2_plot(plot_i)));
                    plot([mean_to_plot mean_to_plot],[0 0.25],"color",[0.6350 0.0780 0.1840],'LineWidth',4,'linestyle',":");
                    text(mean_to_plot+0.3,0.1,num2str(round(mean_to_plot,2)),"color",[0.6350 0.0780 0.1840],"fontsize",20)
                end
                plot([1 1],[0 0.25],'--k','LineWidth',0.5)
           
                ylabel('Relative Frequency')
                xlabel('Probability Ratio')

            end

temp_scen_vars_to_pull = [1 2 3 5 6 8];

for dest_fire_i = 1:length(most_destructive_fires_names)

    lat_to_pull = most_destructive_fires_lats(dest_fire_i);
    
    good_fire_inds = find(Ignition_lat == lat_to_pull);

    if ~isempty(good_fire_inds)
        
        all_predictor_variables_fire_of_interest = all_vars_final_day_cmip6_altered(good_fire_inds,:,:);
        Fire_ID_fire_of_interest = Fire_ID(good_fire_inds);
        Date_fire_of_interest = Date(good_fire_inds);
        Ignition_lat_fire_of_interest = Ignition_lat(good_fire_inds);
        Ignition_lon_fire_of_interest = Ignition_lon(good_fire_inds);
        Int_perim_24_fire_of_interest = Int_perim_24(good_fire_inds);
        binary_response_fire_of_interest = binary_response(good_fire_inds);
        all_fire_probabilities_top_models_mean_fire_of_interest = all_fire_probabilities_top_models_mean(good_fire_inds,:);
        all_fire_PRs_top_models_mean_fire_of_interest = all_fire_PRs_top_models_mean(good_fire_inds,:);
        
        horzspace = 0.08;
        vertspace = 0.05;
        
                figure('Position', [100, 100, 650, 800]);
                set(gcf,'color',[1 1 1]);
                set(0, 'DefaultAxesFontSize',10);
                set(0,'defaultAxesFontName', 'helvetica')
                hold on
        
                subaxis(3,2,1,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        
                    plot(Date_fire_of_interest,Int_perim_24_fire_of_interest,'-k','LineWidth',2)
                    plot([Date_fire_of_interest(1) Date_fire_of_interest(end)], [10000 10000],':k')
        
                    ylabel('Daily Growth')
                    title(most_destructive_fires_names{dest_fire_i})

                    set(gca,'xticklabel',[])
        
                subaxis(3,2,3,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        
                    plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,temp_scen_vars_to_pull),'LineWidth',2)
        
                    ylabel('Probability of Extreme daily Growth')
                    set(gca,'xticklabel',[])
        
                subaxis(3,2,5,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        
                    plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,temp_scen_vars_to_pull),'LineWidth',2)
        
                    ylabel('Probability Ratio')
        
                subaxis(3,2,2,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        
                    plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,1,temp_scen_vars_to_pull)),'LineWidth',2)
        
                    ylabel('Temperature (K)')
                    legend(temp_scenario_vars{temp_scen_vars_to_pull},'location','best','Box','off')
                    set(gca,'xticklabel',[])
     
                subaxis(3,2,4,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        
                    plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,temp_scen_vars_to_pull)),'LineWidth',2)
        
                    ylabel('Vapor Pressure Deficit (hPa)')
                    set(gca,'xticklabel',[])
    
                subaxis(3,2,6,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        
                    plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,temp_scen_vars_to_pull)),'-','LineWidth',2)
                    plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,6,temp_scen_vars_to_pull)),':','LineWidth',2)
        
                    ylabel('100-hr or 1000-hr dead fuel moisture (%)')

    end
end

row_num = 3;
col_num = 4;

plot_locs = [1 5 9 2 6 10 3 7 11 4 8 12];

horzspace = 0.01;
vertspace = 0.02;

line_styles = {};
line_colors = {};

        figure('Position', [100, 100, 650, 800]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',10);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

column_top = 1;

fires_to_plot = [1 4 7 8];

for dest_fire_i = 1:length(fires_to_plot)

    lat_to_pull = most_destructive_fires_lats(fires_to_plot(dest_fire_i));
    
    good_fire_inds = find(Ignition_lat == lat_to_pull);

    if ~isempty(good_fire_inds)
        
        all_predictor_variables_fire_of_interest = all_vars_final_day_cmip6_altered(good_fire_inds,:,:);
        Fire_ID_fire_of_interest = Fire_ID(good_fire_inds);
        Date_fire_of_interest = Date(good_fire_inds);
        Ignition_lat_fire_of_interest = Ignition_lat(good_fire_inds);
        Ignition_lon_fire_of_interest = Ignition_lon(good_fire_inds);
        Int_perim_24_fire_of_interest = Int_perim_24(good_fire_inds);
        binary_response_fire_of_interest = binary_response(good_fire_inds);
        all_fire_probabilities_top_models_mean_fire_of_interest = all_fire_probabilities_top_models_mean(good_fire_inds,:);
        all_fire_PRs_top_models_mean_fire_of_interest = all_fire_PRs_top_models_mean(good_fire_inds,:);
        
                subaxis(row_num,col_num,column_top,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        
                    plot(Date_fire_of_interest,Int_perim_24_fire_of_interest,'-k','LineWidth',2)
                    scatter(Date_fire_of_interest,Int_perim_24_fire_of_interest,30,"black","filled")
                    scatter(Date_fire_of_interest(binary_response_fire_of_interest == 1),Int_perim_24_fire_of_interest(binary_response_fire_of_interest == 1),30,"magenta","filled")
                    plot([Date_fire_of_interest(1) Date_fire_of_interest(end)], [10000 10000],':k')

                        expected_frequencys = sum(all_fire_probabilities_top_models_mean_fire_of_interest,1);
                        mean_probs = mean(all_fire_probabilities_top_models_mean_fire_of_interest,1);
                        chaces_of_one = (1-(1-mean_probs).^length(Date_fire_of_interest));

                        text(Date_fire_of_interest(3),60000,num2str(round(chaces_of_one(1),2)),"FontSize",12,"Color","blue")
                        text(Date_fire_of_interest(3),66000,num2str(round(chaces_of_one(2),2)),"FontSize",12,"Color","black")
                        text(Date_fire_of_interest(3),75000,num2str(round(chaces_of_one(3),2)),"FontSize",12,"Color",[0.9290 0.6940 0.1250])
                        text(Date_fire_of_interest(3),81000,num2str(round(chaces_of_one(5),2)),"FontSize",12,"Color",[0.6350 0.0780 0.1840])
                        text(Date_fire_of_interest(3),90000,num2str(round(chaces_of_one(6),2)),"FontSize",12,"Color",[0.9290 0.6940 0.1250])
                        text(Date_fire_of_interest(3),96000,num2str(round(chaces_of_one(8),2)),"FontSize",12,"Color",[0.6350 0.0780 0.1840])

                    ylim([0 100000])

                    if column_top == 1; ylabel('Daily Growth'); end
                    if column_top > 1; set(gca,'yticklabel',[]); end

                    title(most_destructive_fires_names{fires_to_plot(dest_fire_i)})

                    set(gca,'xticklabel',[])
        
                subaxis(row_num,col_num,column_top+4,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on

                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,1),'-b','LineWidth',1.5)
                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,2),'-k','LineWidth',1.5)
                    
                             plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,3),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-.')
                             plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,5),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-.')
                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,6),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-')
                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,8),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-')
                
        
                    ylim([0 0.18])
        
                    if column_top == 1; ylabel('Probability of Extreme Daily Growth'); end
                    if column_top > 1; set(gca,'yticklabel',[]); end
                    
                    set(gca,'xticklabel',[])
        
                subaxis(row_num,col_num,column_top+8,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
                            h1 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,1),'-b','LineWidth',1.5);
                            h2 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,2),'-k','LineWidth',1.5);
                    
                            h3 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,3),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-.');
                            h4 =  plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,5),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-.');
                            h5 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,6),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-');
                            h6 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,8),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-');

                    ylim([0.95 13])

                    if column_top > 1; set(gca,'yticklabel',[]); end

                    if column_top == 3    
                        
                        legend([h6 h4 h5 h3 h2 h1],{'Very High Emissions, End-Century',...
                                                    'Very High Emissions, Mid-Century',...
                                                    'Low Emissions, End-Century',...
                                                    'Low Emissions, Mid-Century',...
                                                    'Historical, Predicted',...
                                                    'Preindustrial'},...
                                                    'Location','Best','Box','off','Fontsize',5.5)

                    end

                    a = get(gca,'XTickLabel');  

                    if column_top == 1; ylabel('Probability Ratio',"fontsize",10); end

                    column_top = column_top + 1;
        
    end
end

row_num = 5;
col_num = 2;

horzspace = 0.01;
vertspace = 0.02;

line_styles = {};
line_colors = {};

        figure('Position', [100, 100, 670, 800]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',7);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

column_top = 1;

fires_to_plot = [7 4];

for dest_fire_i = 1:length(fires_to_plot)

    lat_to_pull = most_destructive_fires_lats(fires_to_plot(dest_fire_i));
    
    good_fire_inds = find(Ignition_lat == lat_to_pull);

    if ~isempty(good_fire_inds)
        
        all_predictor_variables_fire_of_interest = all_vars_final_day_cmip6_altered(good_fire_inds,:,:);
        Fire_ID_fire_of_interest = Fire_ID(good_fire_inds);
        Date_fire_of_interest = Date(good_fire_inds);
        Ignition_lat_fire_of_interest = Ignition_lat(good_fire_inds);
        Ignition_lon_fire_of_interest = Ignition_lon(good_fire_inds);
        Int_perim_24_fire_of_interest = Int_perim_24(good_fire_inds);
        binary_response_fire_of_interest = binary_response(good_fire_inds);
        all_fire_probabilities_top_models_mean_fire_of_interest = all_fire_probabilities_top_models_mean(good_fire_inds,:);
        all_fire_PRs_top_models_mean_fire_of_interest = all_fire_PRs_top_models_mean(good_fire_inds,:);
        
                subaxis(row_num,col_num,column_top,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        Int_perim_24_fire_of_interest
                    plot(Date_fire_of_interest,Int_perim_24_fire_of_interest,'-k','LineWidth',2)
                    scatter(Date_fire_of_interest,Int_perim_24_fire_of_interest,30,"black","filled")
                    scatter(Date_fire_of_interest(binary_response_fire_of_interest == 1),Int_perim_24_fire_of_interest(binary_response_fire_of_interest == 1),30,"magenta","filled")
                    plot([Date_fire_of_interest(1) Date_fire_of_interest(end)], [10000 10000],':k')

                    ylim([0 60000])
                    xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])

                    if column_top == 1; ylabel('Daily Growth (acres)',"fontsize",10); end
                    if column_top > 1; set(gca,'yticklabel',[]); end

                    title(most_destructive_fires_names{fires_to_plot(dest_fire_i)},"fontsize",16)

                    set(gca,'xticklabel',[])
        
                subaxis(row_num,col_num,column_top+2,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on

                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,1),'-b','LineWidth',1.5)
                             plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,2),'-k','LineWidth',1.5)
                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,6),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-')
                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,8),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-')
                
        
                    ylim([0 0.18])
                    xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])
        
                    if column_top == 1; ylabel('Probability',"fontsize",10); end
                    if column_top > 1; set(gca,'yticklabel',[]); end
                    
                    set(gca,'xticklabel',[])
        
                subaxis(row_num,col_num,column_top+4,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
                            h1 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,1),'-b','LineWidth',1.5);
                             h2 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,2),'-k','LineWidth',1.5);
                            h5 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,6),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-');
                            h6 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,8),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-');

                    if column_top == 1; ylabel('Probability Ratio',"fontsize",10); end
                    if column_top > 1; set(gca,'yticklabel',[]); end
                    set(gca,'xticklabel',[])

                    ylim([0.95 13])
                    xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])

                subaxis(row_num,col_num,column_top+6,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on

                            h1 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,1)),'-b','LineWidth',1.5);
                             h2 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,2)),'-k','LineWidth',1.5);
                            h5 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,6)),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-');
                            h6 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,8)),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-');

                            plot([min(Date_fire_of_interest) max(Date_fire_of_interest)],[10 10],'--k')
                    
                            xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])
                            ylim([0 15])

                    if column_top == 1; ylabel('100-hr Dead Fuel Moisture (%)',"fontsize",8); end
                    if column_top > 1; set(gca,'yticklabel',[]); end
                    set(gca,'xticklabel',[])

                  subaxis(row_num,col_num,column_top+8,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on

                            h1 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,1)),'-b','LineWidth',1.5);
                             h2 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,2)),'-k','LineWidth',1.5);
                            h5 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,6)),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-');
                            h6 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,8)),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-');

                            plot([min(Date_fire_of_interest) max(Date_fire_of_interest)],[1.5 1.5],'--k')

                            ylim([0 8])
                            xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])

                    if column_top == 1; ylabel('Vapor Pressure Deficit (kPa)',"fontsize",8); end
                    if column_top > 1; set(gca,'yticklabel',[]); end

                    column_top = column_top + 1;
        
    end
end

row_num = 5;
col_num = 4;

horzspace = 0.01;
vertspace = 0.02;

line_styles = {};
line_colors = {};

        figure('Position', [100, 100, 670, 800]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',7);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

column_top = 1;

fires_to_plot = [11 12 14 15];

for dest_fire_i = 1:length(fires_to_plot)

    lat_to_pull = most_destructive_fires_lats(fires_to_plot(dest_fire_i));
    
    good_fire_inds = find(Ignition_lat == lat_to_pull);

    if ~isempty(good_fire_inds)
        
        all_predictor_variables_fire_of_interest = all_vars_final_day_cmip6_altered(good_fire_inds,:,:);
        Fire_ID_fire_of_interest = Fire_ID(good_fire_inds);
        Date_fire_of_interest = Date(good_fire_inds);
        Ignition_lat_fire_of_interest = Ignition_lat(good_fire_inds);
        Ignition_lon_fire_of_interest = Ignition_lon(good_fire_inds);
        Int_perim_24_fire_of_interest = Int_perim_24(good_fire_inds);
        binary_response_fire_of_interest = binary_response(good_fire_inds);
        all_fire_probabilities_top_models_mean_fire_of_interest = all_fire_probabilities_top_models_mean(good_fire_inds,:);
        all_fire_PRs_top_models_mean_fire_of_interest = all_fire_PRs_top_models_mean(good_fire_inds,:);
        
                subaxis(row_num,col_num,column_top,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
        
                    plot(Date_fire_of_interest,Int_perim_24_fire_of_interest,'-k','LineWidth',2)
                    scatter(Date_fire_of_interest,Int_perim_24_fire_of_interest,30,"black","filled")
                    scatter(Date_fire_of_interest(binary_response_fire_of_interest == 1),Int_perim_24_fire_of_interest(binary_response_fire_of_interest == 1),30,"magenta","filled")
                    plot([Date_fire_of_interest(1) Date_fire_of_interest(end)], [10000 10000],':k')
        
                    ylim([0 60000])
                    xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])

                    if column_top == 1; ylabel('Daily Growth',"fontsize",10); end
                    if column_top > 1; set(gca,'yticklabel',[]); end

                    title(most_destructive_fires_names{fires_to_plot(dest_fire_i)},"fontsize",16)

                    set(gca,'xticklabel',[])
        
                subaxis(row_num,col_num,column_top+4,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on

                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,1),'-b','LineWidth',1.5)
                             plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,2),'-k','LineWidth',1.5)
                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,6),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-')
                            plot(Date_fire_of_interest,all_fire_probabilities_top_models_mean_fire_of_interest(:,8),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-')
                
        
                    ylim([0 0.18])
                    xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])
        
                    if column_top == 1; ylabel('Probability',"fontsize",10); end
                    if column_top > 1; set(gca,'yticklabel',[]); end
                    
                    set(gca,'xticklabel',[])
        
                subaxis(row_num,col_num,column_top+8,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on
                            h1 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,1),'-b','LineWidth',1.5);
                             h2 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,2),'-k','LineWidth',1.5);
                            h5 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,6),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-');
                            h6 = plot(Date_fire_of_interest,all_fire_PRs_top_models_mean_fire_of_interest(:,8),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-');

                    if column_top == 1; ylabel('Probability Ratio',"fontsize",10); end
                    if column_top > 1; set(gca,'yticklabel',[]); end
                    set(gca,'xticklabel',[])

                    ylim([0.95 13])
                    xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])

                subaxis(row_num,col_num,column_top+12,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on

                            h1 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,1)),'-b','LineWidth',1.5);
                             h2 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,2)),'-k','LineWidth',1.5);
                            h5 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,6)),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-');
                            h6 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,5,8)),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-');

                            plot([min(Date_fire_of_interest) max(Date_fire_of_interest)],[10 10],'--k')
                    
                            xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])
                            ylim([0 15])

                    if column_top == 1; ylabel('100-hr Dead Fuel Moisture (%)',"fontsize",8); end
                    if column_top > 1; set(gca,'yticklabel',[]); end
                    set(gca,'xticklabel',[])

                  subaxis(row_num,col_num,column_top+16,'SpacingHoriz',horzspace,'SpacingVert',vertspace)
                hold on

                            h1 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,1)),'-b','LineWidth',1.5);
                             h2 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,2)),'-k','LineWidth',1.5);
                            h5 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,6)),'color',[0.9290 0.6940 0.1250],'LineWidth',1.5,LineStyle='-');
                            h6 = plot(Date_fire_of_interest,squeeze(all_predictor_variables_fire_of_interest(:,2,8)),'color',[0.6350 0.0780 0.1840],'LineWidth',1.5,LineStyle='-');

                            plot([min(Date_fire_of_interest) max(Date_fire_of_interest)],[1.5 1.5],'--k')

                            ylim([0 8])
                            xlim([min(Date_fire_of_interest) max(Date_fire_of_interest)])

                    if column_top == 1; ylabel('Vapor Pressure Deficit (kPa)',"fontsize",8); end
                    if column_top > 1; set(gca,'yticklabel',[]); end
                    column_top = column_top + 1;
        
    end
end

FAR_mean = mean(all_fire_FARs_top_models_mean,"omitnan");

FARs_below_thresh_inds = find(all_fire_FARs_top_models_mean < FAR_mean);
FARs_above_thresh_inds = find(all_fire_FARs_top_models_mean >= FAR_mean);

        figure('Position', [100, 100, 600, 600]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',7);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        axesm('robinson',...
        'Frame', 'on',...
        'Grid', 'off',...
        'maplatlim',[min(min(wrf_lat_2d)) max(max(wrf_lat_2d))],...
        'maplonlim',[min(min(wrf_lon_2d)) max(max(wrf_lon_2d))])
        tightmap

        pcolorm(wrf_lat_2d,wrf_lon_2d,background,'LineStyle','none');        
        caxis([0 200])

        scatterm(Ignition_lat(FARs_below_thresh_inds),Ignition_lon(FARs_below_thresh_inds),5,'o','filled','MarkerFaceColor',"blue",'MarkerFaceAlpha',0.8)
        scatterm(Ignition_lat(FARs_above_thresh_inds),Ignition_lon(FARs_above_thresh_inds),5,'o','filled','MarkerFaceColor',"red",'MarkerFaceAlpha',0.8)

        colormap(brewermap([],'BuGn'))
        print(gcf,'/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/Fire/Project/matlab programs/FARs_below_thresh_map','-r1000','-dpdf')

        figure('Position', [100, 100, 600, 600]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',11);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        vars_to_plot = 1:11;

        for vars_i = 1:length(vars_to_plot)
        
            subplot(3,4,vars_i)
            hold on

            num_bins = 20;
            min_bin_boarder = min(squeeze(all_vars_final_day_cmip6_altered(:,vars_i,2)));
            max_bin_boarder = max(squeeze(all_vars_final_day_cmip6_altered(:,vars_i,2)));
            bin_boards = linspace(min_bin_boarder,max_bin_boarder,num_bins);
                        
                        h1 = histogram(squeeze(all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,vars_i,2)),bin_boards,...
                                       'FaceColor','r',...
                                       'EdgeColor','r',...
                                       'EdgeAlpha',0.3,...
                                       'FaceAlpha',0.3,...
                                       'Normalization',...
                                       'probability');
                        h2 = histogram(squeeze(all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,vars_i,2)),bin_boards,...
                                       'FaceColor','b',...
                                       'EdgeColor','b',...
                                       'EdgeAlpha',0.3,...
                                       'FaceAlpha',0.3,...
                                       'Normalization',...
                                       'probability');

                        title(all_predictor_variables_figs{vars_i})
            
                        ylabel('relative fraction')

        end

combos_var_1 = [2 2 5 1 1 1];
combos_var_2 = [5 6 6 5 2 6];

        figure('Position', [100, 100, 600, 800]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',11);
        set(0,'defaultAxesFontName', 'helvetica')

        for var_combo_i = 1:length(combos_var_1)

            subplot(3,2,var_combo_i)
            hold on
    
            fars_above_var_1_preind = all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,combos_var_1(var_combo_i),1);
            fars_above_var_1_historical = all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,combos_var_1(var_combo_i),2);
    
            fars_below_var_1_preind = all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,combos_var_1(var_combo_i),1);
            fars_below_var_1_historical = all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,combos_var_1(var_combo_i),2);
    
            fars_above_var_2_preind = all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,combos_var_2(var_combo_i),1);
            fars_above_var_2_historical = all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,combos_var_2(var_combo_i),2);
    
            fars_below_var_2_preind = all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,combos_var_2(var_combo_i),1);
            fars_below_var_2_historical = all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,combos_var_2(var_combo_i),2);
        
             for connect_is = 1:length(FARs_below_thresh_inds)
    
                 plot([fars_below_var_1_preind(connect_is) fars_below_var_1_historical(connect_is)],[fars_below_var_2_preind(connect_is) fars_below_var_2_historical(connect_is)],'-b')
    
             end
             for connect_is = 1:length(FARs_above_thresh_inds)
    
                 plot([fars_above_var_1_preind(connect_is) fars_above_var_1_historical(connect_is)],[fars_above_var_2_preind(connect_is) fars_above_var_2_historical(connect_is)],'-r')
    
             end

                    scatter(fars_below_var_1_preind,fars_below_var_2_preind,9,'o','filled','MarkerFaceColor',"black",'MarkerFaceAlpha',0.3);
                    h1 = scatter(fars_above_var_1_preind,fars_above_var_2_preind,9,'o','filled','MarkerFaceColor',"black",'MarkerFaceAlpha',0.3);

                    h3 = scatter(fars_below_var_1_historical,fars_below_var_2_historical,9,'o','filled','MarkerFaceColor',"blue",'MarkerFaceAlpha',0.4);
                    h2 = scatter(fars_above_var_1_historical,fars_above_var_2_historical,9,'o','filled','MarkerFaceColor',"red",'MarkerFaceAlpha',0.4);

                    if var_combo_i == 1; legend([h1 h2 h3],{'Preindustrial Value','Historical Value, FAR above 33%','Historical Value, FAR below 33%'},"fontsize",8); legend('boxoff'); end
    
                    xlabel(all_predictor_variables_figs{combos_var_1(var_combo_i)})
                    ylabel(all_predictor_variables_figs{combos_var_2(var_combo_i)})

        end


all_fire_PRs_top_models_mean_current = all_fire_PRs_top_models_mean(:,temp_scenarios_i);

PRs_below_1_inds = find(all_fire_PRs_top_models_mean_current < 1);
PRs_above_1_inds = find(all_fire_PRs_top_models_mean_current >= 1);

temp_cut_off = 291;

all_predictors_PRs_below_1_cold_inds = find(all_fire_PRs_top_models_mean_current < 1 & squeeze(all_vars_final_day_cmip6_altered(:,1,2)) <= temp_cut_off);
all_predictors_PRs_below_1_hot_inds = find(all_fire_PRs_top_models_mean_current < 1 & squeeze(all_vars_final_day_cmip6_altered(:,1,2)) > temp_cut_off);

        figure('Position', [100, 100, 600, 600]); %[left bottom width height]
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',7);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        axesm('robinson',...
        'Frame', 'on',...
        'Grid', 'off',...
        'maplatlim',[min(min(wrf_lat_2d)) max(max(wrf_lat_2d))],...
        'maplonlim',[min(min(wrf_lon_2d)) max(max(wrf_lon_2d))])
        tightmap

        pcolorm(wrf_lat_2d,wrf_lon_2d,background,'LineStyle','none');        
        caxis([0 200])

        scatterm(Ignition_lat(PRs_above_1_inds),Ignition_lon(PRs_above_1_inds),5,'o','filled','MarkerFaceColor',"red",'MarkerFaceAlpha',0.8)
        scatterm(Ignition_lat(PRs_below_1_inds),Ignition_lon(PRs_below_1_inds),5,'o','filled','MarkerFaceColor',"blue",'MarkerFaceAlpha',0.8)

        colormap(brewermap([],'BuGn'))

        figure('Position', [100, 100, 600, 600]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',11);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        vars_to_plot = 1:11;
        temp_scenario_to_plot = 2;

        for vars_i = 1:length(vars_to_plot)
        
            subplot(3,4,vars_i)
            hold on

            num_bins = 20;
            min_bin_boarder = min(squeeze(all_vars_final_day_cmip6_altered(:,vars_i,2)));
            max_bin_boarder = max(squeeze(all_vars_final_day_cmip6_altered(:,vars_i,2)));
            bin_boards = linspace(min_bin_boarder,max_bin_boarder,num_bins);
                        
                        h1 = histogram(squeeze(all_vars_final_day_cmip6_altered(PRs_above_1_inds,vars_i,2)),bin_boards,...
                                       'FaceColor','r',...
                                       'EdgeColor','r',...
                                       'EdgeAlpha',0.3,...
                                       'FaceAlpha',0.3,...
                                       'Normalization',...
                                       'probability');
                        h2 = histogram(squeeze(all_vars_final_day_cmip6_altered(PRs_below_1_inds,vars_i,2)),bin_boards,...
                                       'FaceColor','b',...
                                       'EdgeColor','b',...
                                       'EdgeAlpha',0.3,...
                                       'FaceAlpha',0.3,...
                                       'Normalization',...
                                       'probability');

                        title(all_predictor_variables_figs{vars_i})
            
                        ylabel('relative fraction')

        end

most_destructive_fires_mean_vpds = NaN(length(most_destructive_fires_names),1);
most_destructive_fires_mean_100hr_dfms = NaN(length(most_destructive_fires_names),1);

for dest_fire_i = 1:length(most_destructive_fires_names)

    lat_to_pull = most_destructive_fires_lats(dest_fire_i);
    
    good_fire_inds = find(Ignition_lat == lat_to_pull);

    if ~isempty(good_fire_inds)
        
        all_predictor_variables_fire_of_interest = all_vars_final_day_cmip6_altered(good_fire_inds,:,:);

        most_destructive_fires_mean_vpds(dest_fire_i) = squeeze(mean(mean(all_predictor_variables_fire_of_interest(:,2,1:2),3),1));
        most_destructive_fires_mean_100hr_dfms(dest_fire_i) = squeeze(mean(mean(all_predictor_variables_fire_of_interest(:,5,1:2),3),1));

    end
end

most_destructive_fires_mean_vpds_large_growth_days = NaN(length(most_destructive_fires_names),1);
most_destructive_fires_mean_100hr_dfms_large_growth_days = NaN(length(most_destructive_fires_names),1);

for dest_fire_i = 1:length(most_destructive_fires_names)

    lat_to_pull = most_destructive_fires_lats(dest_fire_i);

    good_fire_inds = find(Ignition_lat == lat_to_pull & binary_response == 1);

    if ~isempty(good_fire_inds)
        
        all_predictor_variables_fire_of_interest = all_vars_final_day_cmip6_altered(good_fire_inds,:,:);

        most_destructive_fires_mean_vpds_large_growth_days(dest_fire_i) = squeeze(mean(mean(all_predictor_variables_fire_of_interest(:,2,1:2),3),1));
        most_destructive_fires_mean_100hr_dfms_large_growth_days(dest_fire_i) = squeeze(mean(mean(all_predictor_variables_fire_of_interest(:,5,1:2),3),1));

    end
end

[rand_prs_dest_fires_vals, rand_prs_dest_fires_is] = sort(array_of_info_for_most_destructive_fires_lifetime_mean(:,3),'descend');
good_destructive_inds_reordered = rand_prs_dest_fires_is(4:end);

mean_PR = mean(all_fire_PRs_top_models_mean_current);

[y,idx3] = datasample(Date,600);

all_fire_PRs_top_models_mean_current_sub = all_fire_PRs_top_models_mean_current(idx3);
all_vars_final_day_cmip6_altered_sub = all_vars_final_day_cmip6_altered(idx3,:,:);

PRs_below_thresh_inds = find(all_fire_PRs_top_models_mean_current_sub < mean_PR);
PRs_above_thresh_inds = find(all_fire_PRs_top_models_mean_current_sub >= mean_PR);

text_100_hr_dfms = linspace(38,14,length(good_destructive_inds_reordered));
text_100_vpds = zeros(length(good_destructive_inds_reordered),1) + 3.7;

combos_var_1 = 2;
combos_var_2 = 5;

        figure('Position', [100, 100, 500, 750]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',13);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        subaxis(2,1,1,'SpacingHoriz',0.05,'SpacingVert',0.04)
            hold on


            PRs_above_var_1_preind = all_vars_final_day_cmip6_altered_sub(PRs_above_thresh_inds,combos_var_1,1);
            PRs_above_var_1_historical = all_vars_final_day_cmip6_altered_sub(PRs_above_thresh_inds,combos_var_1,2);
    
            PRs_below_var_1_preind = all_vars_final_day_cmip6_altered_sub(PRs_below_thresh_inds,combos_var_1,1);
            PRs_below_var_1_historical = all_vars_final_day_cmip6_altered_sub(PRs_below_thresh_inds,combos_var_1,2);
    
            PRs_above_var_2_preind = all_vars_final_day_cmip6_altered_sub(PRs_above_thresh_inds,combos_var_2,1);
            PRs_above_var_2_historical = all_vars_final_day_cmip6_altered_sub(PRs_above_thresh_inds,combos_var_2,2);
    
            PRs_below_var_2_preind = all_vars_final_day_cmip6_altered_sub(PRs_below_thresh_inds,combos_var_2,1);
            PRs_below_var_2_historical = all_vars_final_day_cmip6_altered_sub(PRs_below_thresh_inds,combos_var_2,2);
    
    
             for connect_is = 1:length(PRs_below_thresh_inds)

                   arrow([PRs_below_var_1_preind(connect_is) ...
                     PRs_below_var_2_preind(connect_is)],...
                     [PRs_below_var_1_historical(connect_is) ...
                     PRs_below_var_2_historical(connect_is) ],'color','k','Width',0.4,'Length',4,'BaseAngle',20,'TipAngle',20)

             end
             for connect_is = 1:length(PRs_above_thresh_inds)
    
                 arrow([PRs_above_var_1_preind(connect_is) ...
                     PRs_above_var_2_preind(connect_is)],...
                     [PRs_above_var_1_historical(connect_is) ...
                     PRs_above_var_2_historical(connect_is) ],'color','r','Width',0.4,'Length',4,'BaseAngle',20,'TipAngle',20)

             end

                     for dest_i = 1:length(good_destructive_inds_reordered)

                         if array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),3) > mean_PR;
                              text(text_100_vpds(dest_i),...
                                  text_100_hr_dfms(dest_i),...
                                  strcat(most_destructive_fires_names{good_destructive_inds_reordered(dest_i)},', ',...
                                  num2str(year(day_for_largest_growth_day(good_destructive_inds_reordered(dest_i)))),', ',...
                                  num2str(round(array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),3),2))),...
                                  'color','red','FontSize',11,"FontWeight","bold")
                        
                                  plot([text_100_vpds(dest_i) most_destructive_fires_mean_vpds(good_destructive_inds_reordered(dest_i))],...
                                        [text_100_hr_dfms(dest_i) most_destructive_fires_mean_100hr_dfms(good_destructive_inds_reordered(dest_i))],'-r')
    
                         end
                         if array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),3) <= mean_PR;
                              text(text_100_vpds(dest_i),...
                                  text_100_hr_dfms(dest_i),...
                                  strcat(most_destructive_fires_names{good_destructive_inds_reordered(dest_i)},', ',...
                                  num2str(year(day_for_largest_growth_day(good_destructive_inds_reordered(dest_i)))),', ',...
                                  num2str(round(array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),3),2))),...
                                  'color','black','FontSize',11,"FontWeight","bold")
                        
                                  plot([text_100_vpds(dest_i) most_destructive_fires_mean_vpds(good_destructive_inds_reordered(dest_i))],...
                                        [text_100_hr_dfms(dest_i) most_destructive_fires_mean_100hr_dfms(good_destructive_inds_reordered(dest_i))],'-k')
    
                         end
                     end

                    ylabel('100 Hour Dead Fuel Moisture')
                    title('Probability Ratio')
                    xlim([0 5])
                    ylim([0 40])

                    set(gca,'xticklabel',[])

        [rand_prs_dest_fires_vals, rand_prs_dest_fires_is] = sort(array_of_info_for_most_destructive_fires_lifetime_mean(:,10),'descend');
         good_destructive_inds_reordered = rand_prs_dest_fires_is(4:end);

        subaxis(2,1,2,'SpacingHoriz',0.05,'SpacingVert',0.04)
             hold on

             for connect_is = 1:length(FARs_below_thresh_inds)

                   arrow([fars_below_var_1_preind(connect_is) ...
                     fars_below_var_2_preind(connect_is)],...
                     [fars_below_var_1_historical(connect_is) ...
                     fars_below_var_2_historical(connect_is) ],'color','k','Width',0.4,'Length',4,'BaseAngle',20,'TipAngle',20)

             end
             for connect_is = 1:length(FARs_above_thresh_inds)
    
                 arrow([fars_above_var_1_preind(connect_is) ...
                     fars_above_var_2_preind(connect_is)],...
                     [fars_above_var_1_historical(connect_is) ...
                     fars_above_var_2_historical(connect_is) ],'color','r','Width',0.4,'Length',4,'BaseAngle',20,'TipAngle',20)

             end
    
            fars_above_var_1_preind = all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,combos_var_1,1);
            fars_above_var_1_historical = all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,combos_var_1,2);
    
            fars_below_var_1_preind = all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,combos_var_1,1);
            fars_below_var_1_historical = all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,combos_var_1,2);
    
            fars_above_var_2_preind = all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,combos_var_2,1);
            fars_above_var_2_historical = all_vars_final_day_cmip6_altered(FARs_above_thresh_inds,combos_var_2,2);
    
            fars_below_var_2_preind = all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,combos_var_2,1);
            fars_below_var_2_historical = all_vars_final_day_cmip6_altered(FARs_below_thresh_inds,combos_var_2,2);

                     for dest_i = 1:length(good_destructive_inds_reordered)

                         if array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),10) > FAR_mean
                              text(text_100_vpds(dest_i),...
                                  text_100_hr_dfms(dest_i),...
                                  strcat(most_destructive_fires_names{good_destructive_inds_reordered(dest_i)},', ',...
                                  num2str(year(day_for_largest_growth_day(good_destructive_inds_reordered(dest_i)))),', ',...
                                  num2str(round(100.*array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),10))),'%'),...
                                  'color','red','FontSize',11,"FontWeight","bold")
                        
                                  plot([text_100_vpds(dest_i) most_destructive_fires_mean_vpds_large_growth_days(good_destructive_inds_reordered(dest_i))],...
                                        [text_100_hr_dfms(dest_i) most_destructive_fires_mean_100hr_dfms_large_growth_days(good_destructive_inds_reordered(dest_i))],'-r')
    
                         end
                         if array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),10) <= FAR_mean
                              text(text_100_vpds(dest_i),...
                                  text_100_hr_dfms(dest_i),...
                                  strcat(most_destructive_fires_names{good_destructive_inds_reordered(dest_i)},', ',...
                                  num2str(year(day_for_largest_growth_day(good_destructive_inds_reordered(dest_i)))),', ',...
                                  num2str(round(100.*array_of_info_for_most_destructive_fires_lifetime_mean(good_destructive_inds_reordered(dest_i),10))),'%'),...
                                  'color','black','FontSize',11,"FontWeight","bold")
                        
                                  plot([text_100_vpds(dest_i) most_destructive_fires_mean_vpds_large_growth_days(good_destructive_inds_reordered(dest_i))],...
                                        [text_100_hr_dfms(dest_i) most_destructive_fires_mean_100hr_dfms_large_growth_days(good_destructive_inds_reordered(dest_i))],'-k')
    
                         end
                     end

                    xlabel('Vapor Pressure Deficit (kPa)')
                    ylabel('100 Hour Dead Fuel Moisture')

                    xlim([0 5])
                    ylim([0 40])

                    title('Fraction of Risk Attributable to Warming')

        figure('Position', [100, 100, 600, 600]); %[left bottom width height]
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',7);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        axesm('robinson',...
        'Frame', 'on',...
        'Grid', 'off',...
        'maplatlim',[min(min(wrf_lat_2d)) max(max(wrf_lat_2d))],...
        'maplonlim',[min(min(wrf_lon_2d)) max(max(wrf_lon_2d))])
        tightmap

        pcolorm(wrf_lat_2d,wrf_lon_2d,background,'LineStyle','none');        
        caxis([0 200])

        scatterm(Ignition_lat(all_predictors_PRs_below_1_cold_inds),Ignition_lon(all_predictors_PRs_below_1_cold_inds),15,'o','filled','MarkerFaceColor',"black",'MarkerFaceAlpha',0.8)
        scatterm(Ignition_lat(all_predictors_PRs_below_1_hot_inds),Ignition_lon(all_predictors_PRs_below_1_hot_inds),15,'o','filled','MarkerFaceColor',"red",'MarkerFaceAlpha',0.8)

        colormap(brewermap([],'BuGn'))
        print(gcf,'/Users/patrickbrown/Dropbox/SJSU Weather Climate & Human Systems Lab/Projects/Fire/Project/matlab programs/PRs_below_1_hot_cold_map','-r1000','-dpdf')


        figure('Position', [100, 100, 600, 600]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',11);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        vars_to_plot = 1:11;
        temp_scenario_to_plot = 2;

        for vars_i = 1:length(vars_to_plot)
        
            subplot(3,4,vars_i)
            hold on

            num_bins = 20;
            min_bin_boarder = min(squeeze(all_vars_final_day_cmip6_altered(:,vars_i,2)));
            max_bin_boarder = max(squeeze(all_vars_final_day_cmip6_altered(:,vars_i,2)));
            bin_boards = linspace(min_bin_boarder,max_bin_boarder,num_bins);
                        
                        h1 = histogram(squeeze(all_vars_final_day_cmip6_altered(all_predictors_PRs_below_1_cold_inds,vars_i,2)),bin_boards,...
                                       'FaceColor','k',...
                                       'EdgeColor','k',...
                                       'EdgeAlpha',0.3,...
                                       'FaceAlpha',0.3,...
                                       'Normalization',...
                                       'probability');
                        h2 = histogram(squeeze(all_vars_final_day_cmip6_altered(all_predictors_PRs_below_1_hot_inds,vars_i,2)),bin_boards,...
                                       'FaceColor','r',...
                                       'EdgeColor','r',...
                                       'EdgeAlpha',0.3,...
                                       'FaceAlpha',0.3,...
                                       'Normalization',...
                                       'probability');

                        title(all_predictor_variables_figs{vars_i})
            
                        ylabel('relative fraction')
            
        end


for below_1_PR_i = 1:length(PRs_below_1_inds)
    
        figure('Position', [100, 100, 600, 600]);
        set(gcf,'color',[1 1 1]);
        set(0, 'DefaultAxesFontSize',11);
        set(0,'defaultAxesFontName', 'helvetica')
        hold on

        subaxis(2,1,1)
        hold on
        scatter(1:118,squeeze(all_fire_probabilities_top_models(PRs_below_1_inds(below_1_PR_i),1,:)),20,'blue','filled')
        scatter(1:118,squeeze(all_fire_probabilities_top_models(PRs_below_1_inds(below_1_PR_i),2,:)),20,'red','filled')
        
        plot([1 118],[mean(squeeze(all_fire_probabilities_top_models(PRs_below_1_inds(below_1_PR_i),1,:))) ...
                      mean(squeeze(all_fire_probabilities_top_models(PRs_below_1_inds(below_1_PR_i),1,:)))],'-b')

        plot([1 118],[mean(squeeze(all_fire_probabilities_top_models(PRs_below_1_inds(below_1_PR_i),2,:))) ...
                      mean(squeeze(all_fire_probabilities_top_models(PRs_below_1_inds(below_1_PR_i),2,:)))],'-r')  

        ylabel('Probability of Large Growth')

        subaxis(2,1,2)
        hold on

        scatter(1:118,...
            squeeze(all_fire_probabilities_top_models(PRs_below_1_inds(below_1_PR_i),2,:))./...
            squeeze(all_fire_probabilities_top_models(PRs_below_1_inds(below_1_PR_i),1,:)),...
            20,'black','filled')

        xlabel('Top Models')
        ylabel('PR for this fire-day')

end








