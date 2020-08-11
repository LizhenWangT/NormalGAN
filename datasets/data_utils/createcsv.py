import os
import os.path as osp
import sys


if __name__ == "__main__":
    inputf = sys.argv[1]
    output = sys.argv[2]
    name = ['color', 'depth']
    with open(output, 'a') as fout:
        path = osp.abspath(inputf)
        imagelist = os.listdir(osp.join(path, name[-1]))
        imagelist.sort()
        print(path, len(imagelist))
        for j in imagelist:
            imagename = j
            for i in name:
                if not os.path.isfile(osp.join(path, i, imagename)):
                    print('error ', osp.join(path, i, imagename))
            for i in name:
                fout.write(osp.join(path, i, imagename))
                if i == name[-1]:
                    fout.write('\n')
                else:
                    fout.write(',')


