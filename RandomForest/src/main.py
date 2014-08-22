# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Main
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import sys
import time
import json

from master import Master
import test

def main(clmax, np2c, tree_file_format):
    number_of_tree = 3
    
    master = Master(clmax=clmax, np2c=np2c, max_depth=30, min_bag_size=2)
    master.init_client()

    #stdout_file = 'tree_{}_{}_devie_by_2_evaluation'.format(master.clmax, master.np2c)
    #sys.stdout = open(stdout_file, 'w')

    print('~{ Random Forest Parameter }~')
    print('clmax:', master.clmax)
    print('np2c:', master.np2c)
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
        
        master.save_tree(t, tree_file_format.format(clmax=master.clmax, np2c=master.np2c, index=i))

    tree_information = {
        'clmax': master.clmax,
        'spc': master.np2c,
        'max_depth': master.max_depth,
        'min_bag_size': master.min_bag_size,
        'file_list': file_list
    }

    f = open('main.json', 'w')
    json.dump(tree_information, f, indent=2)
    f.close()

    print('\n{} Tree(s) creation successful'.format(number_of_tree))
    print('Total run time: {} sec'.format(total_run_time))
    print('Avg run time per tree: {} sec'.format(1.0*total_run_time/number_of_tree*1.0))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ', sys.argv[0], ' <clmax> <np2c> <tree_file_format or default:tree_{clmax}_{np2c}_sqrt_2_evaluation_{index}.json>')
        sys.exit(1)
    
    clmax = int(sys.argv[1])
    np2c = int(sys.argv[2])
    tree_file_format = 'tree_{clmax}_{np2c}_sqrt_2_evaluation_{index}.json'
    
    if len(sys.argv) == 3:
        main(clmax, np2c, tree_file_format)
    elif len(sys.argv) == 4:
        tree_file_format = sys.argv[3]
        main(clmax, np2c, tree_file_format)
        
    test.main(clmax, tree_file_format[:-13].format(clmax=clmax, np2c=np2c))