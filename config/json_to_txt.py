import json


def main():
    json_file_path = './facepoints_ct.json'
    txt_file_path = './facepoints_ct.txt'
    f_in = open(json_file_path, 'r')
    f_out = open(txt_file_path, 'w')

    j = json.load(f_in)

    for key in j.keys():
        print(key, j[key])
        line = key + ' ' + str(j[key])[1:-1] + '\n'
        f_out.write(line)

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    main()
