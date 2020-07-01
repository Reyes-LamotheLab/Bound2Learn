%%
function Trackmate_STEP_2 = Trackmate_Analysis_Step2(data) 

    prompt_values = {'Time Interval', 'Proportion Start of Alpha', 'TboundalphaStart', 'TboundbetaStart', 'TbleachStart', 'UpperBoundAlpha', 'UpperBoundBeta', 'Percent Variation Bleach (Decimal)', 'Alpha significance', 'Truncation Point'};
    input_two = inputdlg(prompt_values, 'Enter Values', [1 50; 1 50; 1 50; 1 50; 1 50; 1 50; 1 50; 1 50; 1 50; 1 50], {'1','0.5','3000', '5', '25', '6000', '6000', '0.20','0.5', '3'});
   
    [hratio_data, pValue_ratio_data, bic_data, uMLE_data] = Two_exponential_test (data, str2num(input_two{1}), str2num(input_two{2}), str2num(input_two{3}),str2num(input_two{4}), str2num(input_two{5}), str2num(input_two{6}),...
        str2num(input_two{7}),str2num(input_two{8}), str2num(input_two{9}), str2num(input_two{10}));
    hratio_string = strcat('Hypothesis:', num2str(hratio_data));
    pvalue_string = strcat('Pvalue:', num2str(pValue_ratio_data));
    disp(pValue_ratio_data)
    BIC_string = strcat('Single vs Double BIC:', num2str(bic_data(1)), ',', num2str(bic_data(2)));
    uMLE_string_p = strcat('Estimate of p of Alpha:', num2str(uMLE_data(1))); 
    uMLE_string_alpha = strcat('Estimate of alpha:', num2str(uMLE_data(2)));
    uMLE_string_beta = strcat('Estimate of beta:', num2str(uMLE_data(3)));
    uMLE_string_sam_size = strcat('sample size:', num2str(length(data)));
    msgbox({hratio_string, pvalue_string, BIC_string, uMLE_string_p, uMLE_string_alpha, uMLE_string_beta, uMLE_string_sam_size}, 'Two Exponential Analysis')
%% 



%% 
% boot_options_exp = statset('UseParallel',true);
% pdf_truncexp = @(data,mu) exppdf(data,mu) ./(1-expcdf(truncpt,mu));
% [bootci_EXP, bootstat_EXP]=bootci(10000,{pdf_truncexp,On_time_bound_single_filtered},'type','bca','options',boot_options_exp);
% std(bootstat_EXP)

end
