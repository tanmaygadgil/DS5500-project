Session1 = CompleteSet(1:43,:);
Session2 = CompleteSet(44:86,:);
Session3 = CompleteSet(87:129,:);

gest = 4;
fb = cwtfilterbank(SignalLength=10240,SamplingFrequency=2048,FrequencyLimits=[20 450]);

for participant = 1:43
    data_part_gest = Session3{participant, gest};
    if height(data_part_gest) < 71680
        newRow = zeros(1,size(data_part_gest,2)); 
        data_part_gest = [data_part_gest(1:end, :); newRow];
    end
    data_part_gest(isnan(data_part_gest)) = 0;
    for trial = 1:7
        dataset=[];
        for electrode = 1:12
            data_cwt = data_part_gest(((10240*(trial-1))+1):(10240*trial),electrode);
            wt = cwt(data_cwt,FilterBank=fb);
            dataset = [dataset,{abs(wt)}];
            % disp(dataset);
            % temp=['CWT_Data' filesep 'Participant' num2str(participant) filesep 's1_g' num2str(gest) '_t' num2str(trial) '_we' num2str(electrode) '.csv'];
            % writematrix(wt,temp);
        end
        temp=['CWT_Data' filesep 'Participant' num2str(participant) filesep 's3_g' num2str(gest) '_t' num2str(trial) '.mat'];
        disp(temp);
        save(temp, 'dataset')
    end
end