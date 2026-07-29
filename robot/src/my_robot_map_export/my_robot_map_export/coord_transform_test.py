import json


def world_to_pixel(x, y, resolution, origin_x, origin_y, height):
    px = (x - origin_x) / resolution
    py = (y - origin_y) / resolution
    py_img = height - py
    return px, py_img


def pixel_to_world(px, py_img, resolution, origin_x, origin_y, height):
    py = height - py_img
    x = px * resolution + origin_x
    y = py * resolution + origin_y
    return x, y


def main():
    json_path = '/srv/aria/users/hs/aria/robot/maps_export/aria_map_clean_v1.json'

    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    resolution = meta['resolution']
    width = meta['width']
    height = meta['height']
    origin_x = meta['origin']['x']
    origin_y = meta['origin']['y']

    print('=== MAP META ===')
    print(f'resolution: {resolution}')
    print(f'width: {width}, height: {height}')
    print(f'origin_x: {origin_x}, origin_y: {origin_y}')
    print()

    # sample 1
    x, y = 0.0, 0.0
    px, py_img = world_to_pixel(x, y, resolution, origin_x, origin_y, height)
    print('=== WORLD -> PIXEL ===')
    print(f'world ({x}, {y}) -> pixel ({px:.2f}, {py_img:.2f})')
    print()

    # sample 2
    px2, py_img2 = round(px), round(py_img)
    x2, y2 = pixel_to_world(px2, py_img2, resolution, origin_x, origin_y, height)
    print('=== PIXEL -> WORLD ===')
    print(f'pixel ({px2}, {py_img2}) -> world ({x2:.4f}, {y2:.4f})')


if __name__ == '__main__':
    main()
