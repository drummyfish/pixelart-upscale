from PIL import Image

# implementation of many filters can be found at:
#
# https://code.google.com/archive/p/2dimagefilter/downloads?page=1
#
# a good algorithm:
#
# paper: http://johanneskopf.de/publications/pixelart/paper/pixel.pdf
# demo and resources: http://johanneskopf.de/publications/pixelart/supplementary/index.html
#
#=======================================================================

# helper functions:

# https://en.wikipedia.org/wiki/YUV

def rgb_to_yuv(pixel_rgb):
  y = 0.299 * pixel_rgb[0] + 0.587 * pixel_rgb[1] + 0.114 * pixel_rgb[2]
  return (y, 0.492 * (pixel_rgb[2] - y), 0.877 * (pixel_rgb[0] - y))

def compare_pixels_exact(pixel1, pixel2):
  return pixel1[0] == pixel2[0] and pixel1[1] == pixel2[1] and pixel1[2] == pixel2[2]

def compare_pixels_yuv(pixel1, pixel2, thresh_y = 0.2, thresh_u = 0.5, thresh_v = 0.5):
  pixel1_yuv = rgb_to_yuv(pixel1)
  pixel2_yuv = rgb_to_yuv(pixel2)
  return 1 if abs(pixel1_yuv[0] - pixel2_yuv[0]) < thresh_y and abs(pixel1_yuv[1] - pixel2_yuv[1]) < thresh_u and abs(pixel1_yuv[2] - pixel2_yuv[2]) < thresh_v else 0

def add_pixels(pixel_list):
  result = [0,0,0]

  for pixel in pixel_list:
    for i in range(3):
      result[i] += pixel[i]

  return tuple(result)

def divide_pixel(pixel, value):
  return (pixel[0] / value,pixel[1] / value,pixel[2] / value)

def mix_pixels(pixel_list):
  result = [0,0,0]

  for pixel in pixel_list:
    for i in range(3):
      result[i] += pixel[i]

  for i in range(3):
    result[i] /= len(pixel_list)

  return tuple(result)

def upscale_n_times(image, n, upscale_function, neighbour_size):
  def saturate(value, limit_from, limit_to):
    return min(max(limit_from,value),limit_to)

  image = image.convert("RGB")
  width, height = image.size

  source_pixels = image.load()

  result = Image.new("RGB",(n * width,n * height),"white")
  result_pixels = result.load()

  for y in range(height):
    for x in range(width):
      neighbourhood = []
      neighbour_indices = [i for i in range(neighbour_size + 1)] + [i for i in range(-1 * neighbour_size - 1,0)]

      for i in neighbour_indices:
        neighbourhood.append([])

        for j in neighbour_indices:
          neighbourhood[-1].append(source_pixels[(saturate(x + i,0,width - 1),saturate(y + j,0,height - 1))])

      new_pixels = upscale_function(neighbourhood,(x,y))

      for i in range(n):
        for j in range(n):
          result_pixels[(n * x + i,n * y + j)] = new_pixels[j][i]

  return result

#=========================================

def nearest_neighbour_2x(image):
  def func(pixels, coords):
    return (
      (pixels[0][0], pixels[0][0]),
      (pixels[0][0], pixels[0][0])
      )

  return upscale_n_times(image,2,func,1)

def linear_2x(image):
  def func(pixels, coords):
    return (
        (
          mix_pixels([ pixels[-1][-1],pixels[0][-1],pixels[-1][0],pixels[0][0] ]),
          mix_pixels([ pixels[0][-1],pixels[1][-1],pixels[0][0],pixels[1][0] ])
        ),
        ( mix_pixels([ pixels[-1][0],pixels[0][0],pixels[-1][1],pixels[0][1] ]),
          mix_pixels([ pixels[0][0],pixels[1][0],pixels[0][1],pixels[1][1] ])
        )
      )

  return upscale_n_times(image,2,func,1)

def lines_2x(image):
  def func(pixels, coords):
    return (
      (pixels[0][0], pixels[0][0]  ),
      ((0,0,0), (0,0,0) )
      )

  return upscale_n_times(image,2,func,1)

# https://github.com/amadvance/scale2x/blob/master/scale2x.c
# https://en.wikipedia.org/wiki/Pixel_art_scaling_algorithms#EPX.2FScale2.d.97.2FAdvMAME2.d.97

def scale_2x(image):
  def func(pixels, coords):
    if not compare_pixels_exact(pixels[0][-1],pixels[0][1]) and not compare_pixels_exact(pixels[-1][0],pixels[1][0]):
      return (
          (
            pixels[-1][0] if compare_pixels_exact(pixels[-1][0],pixels[0][-1]) else pixels[0][0],
            pixels[1][0]  if compare_pixels_exact(pixels[1][0],pixels[0][-1]) else pixels[0][0]
          ),
          (
            pixels[-1][0] if compare_pixels_exact(pixels[-1][0],pixels[0][1]) else pixels[0][0],
            pixels[1][0]  if compare_pixels_exact(pixels[1][0],pixels[0][1]) else pixels[0][0]
          )
        )

    return (
        (pixels[0][0], pixels[0][0]),
        (pixels[0][0], pixels[0][0])
      )

  return upscale_n_times(image,2,func,1)

# https://github.com/amadvance/scale2x/blob/master/scale3x.c
# https://en.wikipedia.org/wiki/Pixel_art_scaling_algorithms#EPX.2FScale2.d.97.2FAdvMAME2.d.97

def scale_3x(image):
  def func(pixels, coords):
    if not compare_pixels_exact(pixels[0][-1],pixels[0][1]) and not compare_pixels_exact(pixels[-1][0],pixels[1][0]):
      return (
          (
            pixels[-1][0] if compare_pixels_exact(pixels[-1][0],pixels[0][-1]) else pixels[0][0],
            pixels[0][-1] if ( compare_pixels_exact(pixels[-1][0],pixels[0][-1]) and not compare_pixels_exact(pixels[0][0],pixels[1][-1]) ) or ( compare_pixels_exact(pixels[1][0],pixels[0][-1]) and not compare_pixels_exact(pixels[0][0],pixels[-1][-1]) ) else pixels[0][0],
            pixels[1][0]  if compare_pixels_exact(pixels[1][0],pixels[0][-1]) else pixels[0][0]
          ),
          (
            pixels[-1][0] if ( compare_pixels_exact(pixels[-1][0],pixels[0][-1]) and not compare_pixels_exact(pixels[0][0],pixels[-1][1]) ) or ( compare_pixels_exact(pixels[-1][0],pixels[0][1]) and not compare_pixels_exact(pixels[0][0],pixels[-1][-1]) ) else pixels[0][0],
            pixels[0][0],
            pixels[1][0] if ( compare_pixels_exact(pixels[1][0],pixels[0][-1]) and not compare_pixels_exact(pixels[0][0],pixels[1][1]) ) or ( compare_pixels_exact(pixels[1][0],pixels[0][1]) and not compare_pixels_exact(pixels[0][0],pixels[1][-1]) ) else pixels[0][0]
          ),
          (
            pixels[-1][0] if compare_pixels_exact(pixels[-1][0],pixels[0][1]) else pixels[0][0],
            pixels[0][1] if ( compare_pixels_exact(pixels[-1][0],pixels[0][1]) and not compare_pixels_exact(pixels[0][0],pixels[1][1]) ) or ( compare_pixels_exact(pixels[1][0],pixels[0][1]) and not compare_pixels_exact(pixels[0][0],pixels[-1][1]) ) else pixels[0][0],
            pixels[1][0] if compare_pixels_exact(pixels[1][0],pixels[0][1]) else pixels[0][0]
          )
        )

    return (
        (pixels[0][0], pixels[0][0], pixels[0][0]),
        (pixels[0][0], pixels[0][0], pixels[0][0]),
        (pixels[0][0], pixels[0][0], pixels[0][0])
      )

  return upscale_n_times(image,3,func,1)

# TODO: scale_4x

# https://en.wikipedia.org/wiki/Pixel_art_scaling_algorithms#Eagle

def eagle_2x(image):
  def func(pixels, coords):
    return (
        (
          pixels[-1][-1] if compare_pixels_exact( pixels[-1][0], pixels[-1][-1] ) and compare_pixels_exact( pixels[-1][-1], pixels[0][-1] ) else pixels[0][0],
          pixels[1][-1]  if compare_pixels_exact( pixels[1][0], pixels[1][-1] ) and compare_pixels_exact( pixels[1][-1], pixels[0][-1] ) else pixels[0][0]
        ),
        ( 
          pixels[-1][1]  if compare_pixels_exact( pixels[-1][0], pixels[-1][1] ) and compare_pixels_exact( pixels[-1][1], pixels[0][1] ) else pixels[0][0],
          pixels[1][1]   if compare_pixels_exact( pixels[1][0], pixels[1][1] ) and compare_pixels_exact( pixels[1][1], pixels[0][1] ) else pixels[0][0]
        )
      )

  return upscale_n_times(image,2,func,1)

# eagle x3 algorithm found at
# https://code.google.com/archive/p/2dimagefilter/source/default/source?page=2

def eagle_3x(image):
  def func(pixels, coords):
    a = pixels[-1][-1]
    b = pixels[0][-1]
    c = pixels[1][-1]
    d = pixels[-1][0]
    e = pixels[0][0]
    f = pixels[1][0]
    g = pixels[-1][1]
    h = pixels[0][1]
    i = pixels[1][1]

    return (
        (
          mix_pixels([a,b,d]) if compare_pixels_exact(a,b) and compare_pixels_exact(a,d) else e,
          mix_pixels([ mix_pixels([a,b,d]), mix_pixels([c,b,f]) ]) if compare_pixels_exact(a,b) and compare_pixels_exact(a,d) and compare_pixels_exact(c,b) and compare_pixels_exact(c,f) else e,
          mix_pixels([c,b,f]) if compare_pixels_exact(c,b) and compare_pixels_exact(c,f) else e
        ),
        (
          mix_pixels([ mix_pixels([a,b,d]), mix_pixels([g,d,h]) ]) if compare_pixels_exact(a,b) and compare_pixels_exact(a,d) and compare_pixels_exact(g,h) and compare_pixels_exact(g,d) else e,
          e,
          mix_pixels([ mix_pixels([c,b,f]), mix_pixels([i,f,h]) ]) if compare_pixels_exact(c,b) and compare_pixels_exact(c,f) and compare_pixels_exact(i,f) and compare_pixels_exact(i,h) else e
        ),
        (

          mix_pixels([g,d,h]) if compare_pixels_exact(g,d) and compare_pixels_exact(g,h) else e,
          mix_pixels([ mix_pixels([g,h,d]), mix_pixels([i,f,h]) ]) if compare_pixels_exact(g,h) and compare_pixels_exact(g,d) and compare_pixels_exact(i,f) and compare_pixels_exact(i,h) else e,
          mix_pixels([i,f,h]) if compare_pixels_exact(i,f) and compare_pixels_exact(i,h) else e
        )
      )

  return upscale_n_times(image,3,func,1)

# https://news.ycombinator.com/item?id=7925671
# http://pastebin.com/raw/DsNupdbc
# https://github.com/Arcnor/hqx-java/blob/master/src/hqx/Hqx_2x.java

def hq_2x(image):
  hq_table = (
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 15, 12, 5,  3, 17, 13,
      4, 4, 6, 18, 4, 4, 6, 18, 5,  3, 12, 12, 5,  3,  1, 12,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 17, 13, 5,  3, 16, 14,
      4, 4, 6, 18, 4, 4, 6, 18, 5,  3, 16, 12, 5,  3,  1, 14,
      4, 4, 6,  2, 4, 4, 6,  2, 5, 19, 12, 12, 5, 19, 16, 12,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 16, 12, 5,  3, 16, 12,
      4, 4, 6,  2, 4, 4, 6,  2, 5, 19,  1, 12, 5, 19,  1, 14,
      4, 4, 6,  2, 4, 4, 6, 18, 5,  3, 16, 12, 5, 19,  1, 14,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 15, 12, 5,  3, 17, 13,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 16, 12, 5,  3, 16, 12,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 17, 13, 5,  3, 16, 14,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 16, 13, 5,  3,  1, 14,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 16, 12, 5,  3, 16, 13,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 16, 12, 5,  3,  1, 12,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3, 16, 12, 5,  3,  1, 14,
      4, 4, 6,  2, 4, 4, 6,  2, 5,  3,  1, 12, 5,  3,  1, 14
    )

  def helper_blend(rule, e, a, b, d, f, h):
    if rule == 0:
      return e
    elif rule == 1:
      return mix_pixels([e,e,e,a])
    elif rule == 2:
      return mix_pixels([e,e,e,d])
    elif rule == 3:
      return mix_pixels([e,e,e,b])
    elif rule == 4:
      return mix_pixels([e,e,d,b])
    elif rule == 5:
      return mix_pixels([e,e,a,b])
    elif rule == 6:
      return mix_pixels([e,e,a,d])
    elif rule == 7:
      return mix_pixels([e,e,e,e,e,b,b,d])
    elif rule == 8:
      return mix_pixels([e,e,e,e,e,d,d,b])
    elif rule == 9:
      return mix_pixels([e,e,e,e,e,e,d,b])
    elif rule == 10:
      return mix_pixels([e,e,d,d,d,b,b,b])
    elif rule == 11:
      return mix_pixels([e,e,e,e,e,e,e,e,e,e,e,e,e,e,d,b])
    elif rule == 12:
      return mix_pixels([e,e,d,b]) if compare_pixels_yuv(b,d) else e
    elif rule == 13:
      return mix_pixels([e,e,d,d,d,b,b,b]) if compare_pixels_yuv(b,d) else e
    elif rule == 14:
      return mix_pixels([e,e,e,e,e,e,e,e,e,e,e,e,e,e,d,b]) if compare_pixels_yuv(b,d) else e
    elif rule == 15:
      return mix_pixels([e,e,d,b]) if compare_pixels_yuv(b,d) else mix_pixels([e,e,e,a])
    elif rule == 16:
      return mix_pixels([e,e,e,e,e,e,d,b]) if compare_pixels_yuv(b,d) else mix_pixels([e,e,e,a])
    elif rule == 17:
      return mix_pixels([e,e,d,d,d,b,b,b]) if compare_pixels_yuv(b,d) else mix_pixels([e,e,e,a])
    elif rule == 18:
      return mix_pixels([e,e,e,e,e,b,b,d]) if compare_pixels_yuv(b,f) else mix_pixels([e,e,e,d])
    else:
      return mix_pixels([e,e,e,e,e,d,d,b]) if compare_pixels_yuv(d,h) else mix_pixels([e,e,e,b])

  def rotate_pattern(pattern):
    return (pattern[2], pattern[4], pattern[7], pattern[1], pattern[6], pattern[0], pattern[3], pattern[5])

  def pattern_to_number(pattern):
    result = 0
    order = 1

    for bit in reversed(pattern):
      result += order * bit
      order *= 2

    return result

  def func(pixels, coords):
    a = pixels[-1][-1]
    b = pixels[0][-1]
    c = pixels[1][-1]
    d = pixels[-1][0]
    e = pixels[0][0]
    f = pixels[1][0]
    g = pixels[-1][1]
    h = pixels[0][1]
    i = pixels[1][1]

    pattern = (
      not compare_pixels_yuv(e,i),
      not compare_pixels_yuv(e,h),
      not compare_pixels_yuv(e,g),
      not compare_pixels_yuv(e,f),
      not compare_pixels_yuv(e,d),
      not compare_pixels_yuv(e,c),
      not compare_pixels_yuv(e,b),
      not compare_pixels_yuv(e,a),
      )

    p0 = helper_blend(hq_table[pattern_to_number(pattern)], e, a, b, d, f, h)
    pattern = rotate_pattern(pattern)
    p1 = helper_blend(hq_table[pattern_to_number(pattern)], e, c, f, b, h, d)
    pattern = rotate_pattern(pattern)
    p3 = helper_blend(hq_table[pattern_to_number(pattern)], e, i, h, f, d, b)
    pattern = rotate_pattern(pattern)
    p2 = helper_blend(hq_table[pattern_to_number(pattern)], e, g, d, h, b, f)

    return ( (p0,p1), (p2,p3) )

  return upscale_n_times(image,2,func,1)

#---------------------------------------

image = Image.open("test.png")

nearest_neighbour_2x(image).save("nearest.png","PNG")
linear_2x(image).save("linear.png","PNG")
lines_2x(image).save("lines.png","PNG")
eagle_2x(image).save("eagle.png","PNG")
eagle_3x(image).save("eagle3.png","PNG")
scale_2x(image).save("scale_2x.png","PNG")
scale_3x(image).save("scale_3x.png","PNG")
hq_2x(image).save("hq2x.png","PNG")