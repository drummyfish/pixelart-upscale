from PIL import Image
import random

# Implementation of various upscaling pixel art filters - slow, just for
# comparison, not real-time use. Only PIL is required.
#
# Miloslav Ciz 2017, WTFPL license
#
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

# https://en.wikipedia.org/wiki/YUV

def rgb_to_yuv(pixel_rgb):
  pixel_rgb = (pixel_rgb[0] / float(255.0),pixel_rgb[1] / float(255.0),pixel_rgb[2] / float(255.0) )

  y = 0.299 * pixel_rgb[0] + 0.587 * pixel_rgb[1] + 0.114 * pixel_rgb[2]
  return (y, 0.492 * (pixel_rgb[2] - y), 0.877 * (pixel_rgb[0] - y))

#----------------------------------------------------------------------------

def compare_pixels_exact(pixel1, pixel2):
  return pixel1[0] == pixel2[0] and pixel1[1] == pixel2[1] and pixel1[2] == pixel2[2]

#----------------------------------------------------------------------------

def compare_pixels_yuv(pixel1, pixel2, thresh_y = 0.2, thresh_u = 0.2, thresh_v = 0.2):
  pixel1_yuv = rgb_to_yuv(pixel1)
  pixel2_yuv = rgb_to_yuv(pixel2)
  return 1 if abs(pixel1_yuv[0] - pixel2_yuv[0]) < thresh_y and abs(pixel1_yuv[1] - pixel2_yuv[1]) < thresh_u and abs(pixel1_yuv[2] - pixel2_yuv[2]) < thresh_v else 0

#----------------------------------------------------------------------------

def pixelwise_combine(images, combine_function):
  width, height = images[0].size

  result = Image.new("RGB",(width,height),"white")
  result_pixels = result.load()  

  pixels = []
  
  for image in images:
    pixels.append(image.load())

  for y in range(height):
    for x in range(width):
      pixel_values = []

      for pixel_array in pixels:
        pixel_values.append(pixel_array[(x,y)])

      result_pixels[(x,y)] = combine_function(pixel_values)

  return result

#----------------------------------------------------------------------------

def add_pixels(pixel_list):
  result = [0,0,0]

  for pixel in pixel_list:
    for i in range(3):
      result[i] += pixel[i]

  return tuple(result)

#----------------------------------------------------------------------------

def divide_pixel(pixel, value):
  return (pixel[0] / value,pixel[1] / value,pixel[2] / value)

#----------------------------------------------------------------------------

def mix_pixels(pixel_list):
  result = [0,0,0]

  for pixel in pixel_list:
    for i in range(3):
      result[i] += pixel[i]

  for i in range(3):
    result[i] /= len(pixel_list)

  return tuple(result)

#----------------------------------------------------------------------------

def saturate(value, limit_from, limit_to):
  return min(max(limit_from,value),limit_to)

#----------------------------------------------------------------------------

def get_pixel_neighbours(image_pixels, width, height, x, y, neighbour_size):
  neighbourhood = []
  neighbour_indices = [i for i in range(neighbour_size + 1)] + [i for i in range(-1 * neighbour_size - 1,0)]

  for i in neighbour_indices:
    neighbourhood.append([])

    for j in neighbour_indices:
      neighbourhood[-1].append(image_pixels[(saturate(x + i,0,width - 1),saturate(y + j,0,height - 1))])

  return neighbourhood

#----------------------------------------------------------------------------

def upscale_n_times(image, n, upscale_function, neighbour_size):

  image = image.convert("RGB")
  width, height = image.size

  source_pixels = image.load()

  result = Image.new("RGB",(n * width,n * height),"white")
  result_pixels = result.load()

  for y in range(height):
    for x in range(width):
      neighbourhood = get_pixel_neighbours(source_pixels, width, height, x, y, neighbour_size)
      new_pixels = upscale_function(neighbourhood,(x,y))

      for i in range(n):
        for j in range(n):
          result_pixels[(n * x + i,n * y + j)] = new_pixels[j][i]

  return result

#----------------------------------------------------------------------------

def neighbour_pixels_to_letters(pixels):
  return ( pixels[-1][-1], pixels[0][-1], pixels[1][-1], pixels[-1][0], pixels[0][0], pixels[1][0], pixels[-1][1], pixels[0][1], pixels[1][1] )

#============================================================================

def average_image(images):
  return pixelwise_combine(images,mix_pixels)

#----------------------------------------------------------------------------

def random_image(images):
  def random_pixel(pixels):
    return random.choice(pixels)

  return pixelwise_combine(images,random_pixel)

#============================================================================

def nearest_neighbour_2x(image):
  def func(pixels, coords):
    return (
      (pixels[0][0], pixels[0][0]),
      (pixels[0][0], pixels[0][0])
      )

  return upscale_n_times(image,2,func,1)

#----------------------------------------------------------------------------

def nearest_neighbour_3x(image):
  def func(pixels, coords):
    return (
      (pixels[0][0], pixels[0][0], pixels[0][0]),
      (pixels[0][0], pixels[0][0], pixels[0][0]),
      (pixels[0][0], pixels[0][0], pixels[0][0])
      )

  return upscale_n_times(image,3,func,1)

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

def linear_3x(image):
  def func(pixels, coords):
    a, b, c, d, e, f, g, h, i = neighbour_pixels_to_letters(pixels)

    abb = mix_pixels([a,b,b])
    bbc = mix_pixels([b,b,c])

    ghh = mix_pixels([g,h,h])
    hhi = mix_pixels([h,h,i])

    bee = mix_pixels([b,e,e])
    dee = mix_pixels([d,e,e])
    eef = mix_pixels([e,e,f])
    eeh = mix_pixels([e,e,h])

    return (
        (
          mix_pixels([abb,dee,dee]),
          bee,
          mix_pixels([bbc,eef,eef])
        ),
        (
          dee,
          e,
          eef
        ),
        (
          mix_pixels([dee,dee,ghh]),
          eeh,
          mix_pixels([eef,eef,hhi])
        )
      )

  return upscale_n_times(image,3,func,1)

#----------------------------------------------------------------------------

def lines_2x(image):
  def func(pixels, coords):
    return (
      (pixels[0][0], pixels[0][0]),
      ((0,0,0), (0,0,0))
      )

  return upscale_n_times(image,2,func,1)

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

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

#----------------------------------------------------------------------------

# eagle x3 doesn't officially exist, the algorithm was made by the
# author of
# https://code.google.com/archive/p/2dimagefilter/source/default/source?page=2

def eagle_3x(image):
  def func(pixels, coords):
    a, b, c, d, e, f, g, h, i = neighbour_pixels_to_letters(pixels)

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

#----------------------------------------------------------------------------

# another version of eagle x3, made by the
# author of
# https://code.google.com/archive/p/2dimagefilter/source/default/source?page=2

def eagle_3xb(image):
  def func(pixels, coords):
    a, b, c, d, e, f, g, h, i = neighbour_pixels_to_letters(pixels)

    return (
        (
          mix_pixels([a,b,d]) if compare_pixels_exact(a,b) and compare_pixels_exact(a,d) else e,
          e,
          mix_pixels([b,c,f]) if compare_pixels_exact(b,c) and compare_pixels_exact(c,f) else e
        ),
        (
          e,
          e,
          e
        ),
        (
          mix_pixels([d,g,h]) if compare_pixels_exact(d,g) and compare_pixels_exact(g,h) else e,
          e,
          mix_pixels([f,h,i]) if compare_pixels_exact(f,h) and compare_pixels_exact(h,i) else e 
        )
      )

  return upscale_n_times(image,3,func,1)

#----------------------------------------------------------------------------

# https://code.google.com/archive/p/2dimagefilter/source/default/source?page=2

def epx_2x(image):
  
  def helper_condition(v0, v1, v2, v3, v4, v5, v6):
    return compare_pixels_exact(v0,v1) and (
      not compare_pixels_exact(v2,v3) or
      not compare_pixels_exact(v2,v4) or
      not compare_pixels_exact(v0,v5) or
      not compare_pixels_exact(v1,v6) )

  def func(pixels, coords):
    a, b, c, d, e, f, g, h, i = neighbour_pixels_to_letters(pixels)

    p0 = e
    p1 = e
    p2 = e
    p3 = e

    if not compare_pixels_exact(d,f) and not compare_pixels_exact(b,h) and (
      compare_pixels_exact(e,d) or
      compare_pixels_exact(e,h) or
      compare_pixels_exact(e,f) or
      compare_pixels_exact(e,b) or (
        ( not compare_pixels_exact(a,i) or compare_pixels_exact(e,g) or compare_pixels_exact(d,c) ) and
        ( not compare_pixels_exact(g,c) or compare_pixels_exact(e,a) or compare_pixels_exact(e,i) )
        )
      ):

      if helper_condition(b,d,e,a,i,c,g):
        p0 = mix_pixels([b,d])

      if helper_condition(f,b,e,c,g,i,a):
        p1 = mix_pixels([f,b])

      if helper_condition(d,h,e,g,c,a,i):
        p2 = mix_pixels([d,h])

      if helper_condition(h,f,e,i,a,g,c):
        p3 = mix_pixels([h,f])

    return ( (p0, p1), (p2, p3) )

  return upscale_n_times(image,2,func,1)

#----------------------------------------------------------------------------

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
    a, b, c, d, e, f, g, h, i = neighbour_pixels_to_letters(pixels)

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

#----------------------------------------------------------------------------

def experiment_a_3x(image):

  def func(pixels, coords):
    a, b, c, d, e, f, g, h, i = neighbour_pixels_to_letters(pixels)

    return (
        (
          b if compare_pixels_exact(b,d) else e,
          e,
          e if compare_pixels_exact(e,f) or compare_pixels_exact(e,b) else c
        ),
        (
          e,
          e,
          e
        ),
        (
          e,
          e,
          f if compare_pixels_exact(f,h) else e
        )
      )

  return upscale_n_times(image,3,func,1)

#============================================================================

def do_upscale(what, save_as_filename):
  print("computing " + save_as_filename)
  what.save(save_as_filename + ".png","PNG")
  return what
  
image = Image.open("test.png")
random.seed(0)

# basic algorithms:

result_nn_2x       = do_upscale(nearest_neighbour_2x(image),"2x nearest neighbour")
result_linear_2x   = do_upscale(linear_2x(image),"2x linear")
result_lines_2x    = do_upscale(lines_2x(image),"2x lines")
result_eagle_2x    = do_upscale(eagle_2x(image),"2x eagle")
result_scale_2x    = do_upscale(scale_2x(image),"2x scale")
result_hq_2x       = do_upscale(hq_2x(image),"2x hq")
result_epx_2x      = do_upscale(epx_2x(image),"2x epx")

result_nn_3x       = do_upscale(nearest_neighbour_3x(image),"3x nearest neighbour")
result_linear_3x   = do_upscale(linear_3x(image),"3x linear")
result_eagle_3x    = do_upscale(eagle_3x(image),"3x eagle")
result_eagle_3xb   = do_upscale(eagle_3xb(image),"3x eagle b")
result_scale_3x    = do_upscale(scale_3x(image),"3x scale")

result_exp_a       = do_upscale(experiment_a_3x(image),"3x experimental a")

# combines:
result_avg_eagle_scale_hq_epx_2x = average_image([result_eagle_2x,result_scale_2x,result_hq_2x,result_epx_2x])
result_avg_eagle_scale_hq_epx_2x.save("eagle_scale_hq_epx_avg.png","PNG")

result_rnd_eagle_scale_hq_epx_2x = random_image([result_eagle_2x,result_scale_2x,result_hq_2x,result_epx_2x])
result_rnd_eagle_scale_hq_epx_2x.save("eagle_scale_hq_epx_rnd.png","PNG")
