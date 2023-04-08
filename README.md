We provide source code of BEAGLE in this folder.

--- checkpoints: pre-trained stylegan_v2 from https://github.com/NVlabs/stylegan2

--- trojai_round3: application of BEAGLE on TrojAI competition round3
    gen_polygon.py:     attack decomposition on polygon attacked models
                        visualization and decomposed trigger will be saved in 'forensic/trojai_polygon/'

    gen_instagram.py:   attack decomposition on instagram filter attacked models
                        visualization and decomposed trigger will be saved in 'forensic/trojai_instagram/'
    
    abs_beagle_polygon.py: provide the BEAGLE-enhanced ABS to scan polygon attacked models. (output in 'result.txt')
    abs_beagle_filter.py: provide the BEAGLE-enhanced ABS to scan instagram filter attacked models. (output in 'result.txt')
    synthesis_scanner.py: provide the summarized polygon/filter trigger distributions (polygon size/filter parameters) which help build enhaced scanner for ABS.

--- cifar10: application of BEAGLE on CIFAR-10 dataset
    decomposition.py: provide attack decomposition on CIFAR-10 dataset