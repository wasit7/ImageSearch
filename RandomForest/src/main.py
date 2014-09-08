# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Main
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os.path
import sys
import time
import json

from master import Master
import test

tree_root_folder = 'tree'

def main(mainfile, dataset_file):
    number_of_tree = 3
    
    master = Master(dataset_file, max_depth=30, min_bag_size=2)
    master.init_client()

    print('~{ Random Forest Parameter }~')
    print('max depth:', master.max_depth)
    print('min bag size:', master.min_bag_size)
    
    print('Will create {} tree(s):'.format(number_of_tree))
    
    total_run_time = 0
    file_list = []
    # loop to create trees
    for i in range(number_of_tree):
        print('\nCreating tree {}...'.format(i))
        
        t, tree_filename, time_use = master.create_tree()
        file_list.append(tree_filename)
        
        total_run_time += time_use

    tree_information = {
        'max_depth': master.max_depth,
        'min_bag_size': master.min_bag_size,
        'total_run_time': total_run_time,
        'avg_run_time': total_run_time/(number_of_tree*1.0),
        'file_list': file_list
    }

    with open(os.path.join(tree_root_folder, mainfile), 'w') as f:
        json.dump(tree_information, f, indent=2)

    print('\n{} Tree(s) creation successful'.format(number_of_tree))
    print('Total run time: {} sec'.format(total_run_time))
    print('Avg run time per tree: {} sec'.format(1.0*total_run_time/number_of_tree*1.0))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} <main JSON file name> <dataset file>')
        sys.exit(1)
    
    clmax = 5
    
    main(sys.argv[1], sys.argv[2])

    test.main(clmax, os.path.join(tree_root_folder, sys.argv[1]))