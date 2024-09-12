hosting_features = ['feat_query_count','feat_ip_count','feat_ns_count','feat_isns_domain','feat_soadomain_count','feat_issoa_domain','feat_duration']
lex_features = ['feat_suspicious','feat_length','feat_entropy','feat_faketld','feat_brand_pos','feat_pop_keywords','feat_similar','feat_minus']
ip_features1 = ['subnet', 'asn'] 
ip_features2 = ['feat_ip_duration', 'feat_ip_apexcount', 'feat_ip_query']

node_type = [ # 'feat_label_ben','feat_label_mal','feat_label_unknown',#'feat_label_ignore',
         'feat_domain','feat_ip','feat_subnet']#,'feat_asn','feat_city']

ringer = [['length','entropy','num_subdomains','subdomain_len','has_www','valid_tlds','has_single_subdomain',
          'has_tld_subdomain','digit_ex_subdomains_ratio','underscore_ratio','has_ip','has_digits','vowel_ratio',
          'digit_ratio','alphabet_cardinality','repeated_char_ratio','consec_consonants_ratio','consec_digit_ratio'],
          ['1-gram-mean','1-gram-std','1-gram-median','1-gram-max','1-gram-min','1-gram-lower-q','1-gram-upper-q',
          '2-gram-mean','2-gram-std','2-gram-median','2-gram-max','2-gram-min','2-gram-lower-q','2-gram-upper-q',
          '3-gram-mean','3-gram-std','3-gram-median','3-gram-max','3-gram-min','3-gram-lower-q','3-gram-upper-q']]

feature_set = {'hosting': hosting_features, 
               'lex':lex_features, 
               'ip1':ip_features1 ,
               'ip2':ip_features2,
               'ringer1':ringer[0],
               'ringer2':ringer[1]}

feature_labels = node_type.copy()
for k,v in feature_set.items():
    feature_labels += v

base_features = node_type.copy()
base_features += lex_features
base_features += ip_features1
base_features += ringer[0]
base_features += ringer[1]

hosting_group = node_type.copy()
hosting_group += hosting_features
hosting_group += ip_features1
hosting_group += ip_features2

syn_labels = [f'x{i}' for i in range(48)]

hetero_features = {'domain':hosting_features+lex_features + ringer[0] + ringer[1], 'ip':['subnet','asn']} 