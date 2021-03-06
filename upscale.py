from PIL import Image
import random
import math

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

def multiply_pixel(pixel, value):
  return(int(pixel[0] * value),int(pixel[1] * value),int(pixel[2] * value))

#----------------------------------------------------------------------------

def compare_pixels_exact(pixel1, pixel2):
  return pixel1[0] == pixel2[0] and pixel1[1] == pixel2[1] and pixel1[2] == pixel2[2]

#----------------------------------------------------------------------------

def compare_pixels_yuv(pixel1, pixel2, thresh_y = 0.20, thresh_u = 0.20, thresh_v = 0.20):
  pixel1_yuv = rgb_to_yuv(pixel1)
  pixel2_yuv = rgb_to_yuv(pixel2)
  return 1 if abs(pixel1_yuv[0] - pixel2_yuv[0]) < thresh_y and abs(pixel1_yuv[1] - pixel2_yuv[1]) < thresh_u and abs(pixel1_yuv[2] - pixel2_yuv[2]) < thresh_v else 0

#----------------------------------------------------------------------------

def pixel_brightness(pixel):
  return int( 255.0 * (0.21 * pixel[0] / 255.0 + 0.72 * pixel[1] / 255.0 + 0.07 * pixel[2] / 255.0))

#----------------------------------------------------------------------------

def pixel_is_brighter(pixel1, pixel2):
  return pixel_brightness(pixel1) > pixel_brightness(pixel2)

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

def darkest_2x(image):

  def get_brightest(p1,p2,p3,p4):
    if pixel_is_brighter(p1,p2):
      if pixel_is_brighter(p2,p3):
        return p3
      else: 
        return p2
    else:
      if pixel_is_brighter(p1,p3):
        return p3

    return p1

  def func(pixels, coords):
    a, b, c, d, e, f, g, h, i = neighbour_pixels_to_letters(pixels)

    return (
        (
          get_brightest(a,b,d,e),
          get_brightest(b,c,e,f)
        ),
        (
          get_brightest(d,e,g,h),
          get_brightest(e,f,h,i)
        )
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

#----------------------------------------------------------------------------

def experiment_b(image):
  def pattern_size(pattern):
    return int(math.sqrt(len(pattern)))

  def rotate_pattern(pattern):
    new_pattern = []

    p_size = pattern_size(pattern)

    for j in range(p_size):
      for i in range(p_size):
        new_pattern.append(pattern[j + (p_size - i - 1) * p_size])

    return tuple(new_pattern)

  def compare_with_pattern(neighbourhood, pattern):   # return: (pixel, background_color)
    main_pixel = None
    background_pixels = []

    for pass_no in (0,1):
      i = 0

      for y in (-1,0,1):
        for x in (-1,0,1):

          if pass_no == 0:             # check equality of main pixels
            if pattern[i] == 1:
              if main_pixel == None:
                main_pixel = neighbourhood[x][y]
              elif not compare_pixels_exact(main_pixel,neighbourhood[x][y]):
                return None
          else:                        # check inequalities to main pixels
            if pattern[i] == 2 and compare_pixels_exact(main_pixel,neighbourhood[x][y]):
              return None
            else:
              background_pixels.append(neighbourhood[x][y])
          
          i += 1

    return (main_pixel,mix_pixels(background_pixels))

  def put_pattern(image_pixels, width, height, x, y, pattern, pixel, background):
    i = 0

    p_size = pattern_size(pattern)
    p_half_size = p_size / 2

    for y2 in range(p_size):
      for x2 in range(p_size):
        dst_x = saturate(x + x2 - p_half_size,0,width - 1)
        dst_y = saturate(y + y2 - p_half_size,0,height - 1)

        if pattern[i] == 1:
          image_pixels[(dst_x,dst_y)] = pixel
        elif pattern[i] == 2:
          image_pixels[(dst_x,dst_y)] = background
      
        i += 1


  pattern_a = (            # diagonal line
    1, 2, 2,
    2, 1, 2,
    2, 2, 1
    )

  pattern_a2 = (
    0, 2, 2, 0, 0, 0, 0, 0, 0,
    0, 1, 2, 2, 0, 0, 0, 0, 0,
    0, 2, 1, 2, 2, 0, 0, 0, 0,
    0, 2, 2, 1, 2, 2, 0, 0, 0,
    0, 0, 2, 2, 1, 2, 2, 0, 0,
    0, 0, 0, 2, 2, 1, 2, 2, 0,
    0, 0, 0, 0, 2, 2, 1, 2, 0,
    0, 0, 0, 0, 0, 2, 2, 1, 0,
    0, 0, 0, 0, 0, 0, 2, 2, 0
    )

  pattern_b = (            # straight line
    2, 1, 2,
    2, 1, 2,
    2, 1, 2
    )

  pattern_b2 = (
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0
    )

  pattern_c = (             # dot
    2, 2, 2,
    2, 1, 2,
    2, 2, 2
    )

  pattern_c2 = (
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 2, 2, 0, 0, 0,
    0, 0, 2, 2, 1, 2, 2, 0, 0,
    0, 0, 2, 1, 1, 1, 2, 0, 0,
    0, 0, 2, 2, 1, 2, 2, 0, 0,
    0, 0, 0, 2, 2, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 2, 2, 0, 0, 0, 0, 0, 0
    )


  pattern_d = (           # cross
    2, 1, 2,
    1, 1, 1,
    2, 1, 2
    )

  pattern_d2 = (
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    2, 2, 2, 2, 1, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 1, 2, 2, 2, 2,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0,
    0, 0, 0, 2, 1, 2, 0, 0, 0
    )


  pattern_e = (           # corner
    0, 1, 1,
    1, 2, 2,
    1, 2, 0
    )

  pattern_e2 = (
    2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 1, 1, 1, 1, 1, 2, 2,
    2, 1, 1, 2, 2, 2, 2, 2, 2,
    2, 1, 2, 2, 0, 0, 0, 0, 0,
    2, 1, 2, 0, 0, 0, 0, 0, 0,
    2, 1, 2, 0, 0, 0, 0, 0, 0,
    2, 1, 2, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 0, 0, 0, 0, 0, 0,
    )


  pattern_f = (          # dither
    1, 2, 1,
    2, 1, 2,
    1, 2, 1
    )

  pattern_f2 = (
    1, 2, 1, 2, 1, 2, 1, 2, 1,
    2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1,
    2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1,
    2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1,
    2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1
    )

  pattern_g = (         # weird line
    2, 1, 2,
    2, 1, 2,
    1, 2, 0
    )

  pattern_g2 = (
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 2, 2, 1, 2, 2, 0, 0,
    0, 0, 2, 2, 1, 2, 2, 0, 0,
    0, 2, 2, 1, 2, 2, 0, 0, 0,
    2, 2, 1, 2, 2, 0, 0, 0, 0,
    2, 1, 2, 2, 0, 0, 0, 0, 0,
    2, 1, 2, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0
    )

  pattern_gb = (         # weird line flipped
    2, 1, 2,
    2, 1, 2,
    0, 2, 1
    )

  pattern_gb2 = (
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 2, 2, 1, 2, 2, 0, 0,
    0, 0, 2, 2, 1, 2, 2, 0, 0,
    0, 0, 0, 2, 2, 1, 2, 2, 0,
    0, 0, 0, 0, 2, 2, 1, 2, 2,
    0, 0, 0, 0, 0, 2, 2, 1, 2,
    0, 0, 0, 0, 0, 2, 2, 1, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0
    )


  patterns = [
    (pattern_a, pattern_a2),
    (pattern_b, pattern_b2),
    (rotate_pattern(pattern_a), rotate_pattern(pattern_a2)),
    (rotate_pattern(pattern_b), rotate_pattern(pattern_b2)),
    (pattern_c, pattern_c2),
    (pattern_d, pattern_d2),
    (pattern_e, pattern_e2),
    (rotate_pattern(pattern_e), rotate_pattern(pattern_e2)),
    ( rotate_pattern(rotate_pattern(pattern_e)), rotate_pattern(rotate_pattern(pattern_e2)) ),
    ( rotate_pattern(rotate_pattern(rotate_pattern(pattern_e))), rotate_pattern(rotate_pattern(rotate_pattern(pattern_e2))) ),


    (pattern_f, pattern_f2),

    (pattern_g, pattern_g2),
    ( rotate_pattern(pattern_g), rotate_pattern(pattern_g2) ),

    (pattern_gb, pattern_gb2),
    ( rotate_pattern(pattern_gb), rotate_pattern(pattern_gb2) ),
    ]

  width, height = image.size
  source_pixels = image.load()


  upscale_factor = 3

  #result = Image.new("RGB",(upscale_factor * width,upscale_factor * height),"white")

  result = linear_3x(image)

  result_pixels = result.load()











  for y in range(height):
    for x in range(width):
      neighbours = get_pixel_neighbours(source_pixels, width, height, x, y, 1)
      
      matched = False

      for pattern in patterns:

        comparison = compare_with_pattern(neighbours,pattern[0])

        if comparison != None:
          put_pattern(result_pixels, upscale_factor * width, upscale_factor * height, upscale_factor * x, upscale_factor * y, pattern[1], comparison[0], comparison[1])
          matched = True
          break
      



  return result



#----------------------------------------------------------------------------

# My algorithm that tries to preserve thin lines depending on brightness, the
# decision tree has yet to be optimised.

def experiment_c(image):

  def handle_weird_corner(p0,p1,p2,p3):
    if pixel_is_brighter(p1,p0):
      return mix_pixels([p1,p2,p3])
    else:
      return mix_pixels([p0,p1,p2]) #p0

  def handle_straight_line(p0,p1,p2,p3):
    if pixel_is_brighter(p2,p0):
      return mix_pixels([p2,p3])
    else:
      return mix_pixels([p0,p1])

    """
    if compare_pixels_yuv(p2,p3) and pixel_is_brighter(p2,p0):
      return mix_pixels([p2,p3])
    else:
      return mix_pixels([p0,p1])
    """

  def func(pixels, coords):
    a, b, c, d, e, f, g, h, i = neighbour_pixels_to_letters(pixels)

    # a b c
    # d e f
    # g h i

    p1 = mix_pixels([a,b]) if compare_pixels_yuv(a,b) else (a if pixel_is_brighter(a,b) else b)
    p2 = mix_pixels([a,d]) if compare_pixels_yuv(a,d) else (a if pixel_is_brighter(a,d) else d)

    ae = compare_pixels_yuv(a,e)
    bd = compare_pixels_yuv(b,d)
    ab = compare_pixels_yuv(a,b)
    de = compare_pixels_yuv(d,e)
    ad = compare_pixels_yuv(a,d)
    be = compare_pixels_yuv(b,e)

    if ae and bd and ab:                              # all pixels equal?
      p3 = mix_pixels([a,b,d,e])

    elif ad and de and not ab:                        # weird corner 1
      p3 = handle_weird_corner(b,a,e,d) #mix_pixels([a,d,e])
    elif be and de and not ab:                        # weird corner 2
      p3 = handle_weird_corner(a,b,d,e) #mix_pixels([b,d,e])
    elif ad and ab and not be:                        # weird corner 3
      p3 = handle_weird_corner(e,d,b,a) #mix_pixels([a,b,d])
    elif ab and be and not ad:                        # weird corner 4
      p3 = handle_weird_corner(d,a,e,b) #mix_pixels([a,b,e])

    elif ae and (not bd or pixel_is_brighter(b,a)):   # diagonal line 1?
      p3 = mix_pixels([a,e])
    elif bd and (not ae or pixel_is_brighter(a,b)):   # diagonal line 2?
      p3 = mix_pixels([b,d])

    elif ab:   # horizontal line 1?
      p3 = handle_straight_line(a,b,d,e)
    elif de:   # horizontal line 2?
      p3 = handle_straight_line(d,e,a,b)
    elif ad:   # vertical line 1?
      p3 = handle_straight_line(a,d,b,e)
    elif be:   # vertical line 2?
      p3 = handle_straight_line(b,e,a,d)



    else:
      p3 = mix_pixels([a,b,d,e])

    return ( (a,p1), (p2,p3) )

  return upscale_n_times(image,2,func,1)

#----------------------------------------------------------------------------

def experiment_d(input_image, horizontal):

  def decide_pixel(l,r):

    def fits_pattern(pat,left,right): # 0 - dark, 1 - bright, 2 - dont matter
      both = left + right
      bright = []
      dark = []

      for i in range(6):
        if pat[i] == 0:
          dark.append(i)
        elif pat[i] == 1:
          bright.append(i)

      for b in bright:
        for d in dark:
          if not pixel_is_brighter(both[b],both[d]) and not compare_pixels_exact(both[b],both[d]):
            return False

      return True

    #if fits_pattern( (0,1,2, 2,2,0), row_left, row_right ):
    #  return row_left[0]

    #if fits_pattern( (2,1,0, 0,2,2), row_left, row_right ):
    #  return row_left[2]

    if pixel_is_brighter(l[1],r[1]):
      return l[1]

    return r[1]

  old_size = input_image.size

  if horizontal:
    new_size = (input_image.size[0] * 2, input_image.size[1])
  else:
    new_size = (input_image.size[0], input_image.size[1] * 2)

  result = Image.new("RGB",new_size,"white")

  old_pixels = input_image.load()
  new_pixels = result.load()

  for j in range(old_size[1]):
    for i in range(old_size[0]):
      neigh = get_pixel_neighbours(old_pixels,old_size[0],old_size[1],i,j,1)

      if horizontal:
        left  = (neigh[0][-1], neigh[0][0], neigh[0][1])
        right = (neigh[1][-1],  neigh[1][0],  neigh[1][1])
        pos = (i * 2,j)
        pos2 = (i * 2 + 1,j)
      else:
        left  = (neigh[-1][0], neigh[0][0], neigh[1][0])
        right = (neigh[-1][1],  neigh[0][1],  neigh[1][1])
        pos = (i,j * 2)
        pos2 = (i,j * 2 + 1)

      new_pixels[pos] = old_pixels[(i,j)]


      new_pixels[pos2] = decide_pixel(left,right)

  return result

#----------------------------------------------------------------------------

def experiment_f(image):

  def fits_pattern(pat,left,right): # 0 - dark, 1 - bright, 2 - dont matter
    both = left + right
    bright = []
    dark = []

    for i in range(6):
      if pat[i] == 0:
        dark.append(i)
      elif pat[i] == 1:
        bright.append(i)

    for b in bright:
      for d in dark:
        if not pixel_is_brighter(both[b],both[d]) and not compare_pixels_exact(both[b],both[d]):
          return False

    return True

  def decide_pixel(l,r):
    if fits_pattern( (2,1,2,2,0,2), l,r):
      return l[1]

    return r[1]

  def func(pixels, coords):
    p = pixels

    return (
        (
          p[0][0],
          decide_pixel((p[0][-1],p[0][0],p[0][1]),(p[1][-1],p[1][0],p[1][1]))
        ),
        (
          decide_pixel((p[-1][0],p[0][0],p[1][0]),(p[-1][1],p[0][1],p[1][1])),

          mix_pixels([
            decide_pixel(
              (p[0][-1],p[1][0],p[2][1]),
              (p[-1][0],p[0][1],p[1][2])
            ),

           decide_pixel(
               (p[-1][1],p[0][0],p[1][-1]),
               (p[0][2],p[1][1],p[2][0])
            )])


        )
      )

  return upscale_n_times(image,2,func,3)

#============================================================================

def do_upscale(what, save_as_filename):
  print("computing " + save_as_filename)
  what.save(save_as_filename + ".png","PNG")
  return what
  
image = Image.open("test.png")

#h = experiment_d(image,True)
#h.save("out_a.png","PNG")
#v = experiment_d(h,False)
#v.save("out_b.png","PNG")

#experiment_f(image).save("out_c.png","PNG")

darkest_2x(image).save("2x darkest.png","PNG")

random.seed(0)

"""
aaa = experiment_b(image)
ccc = linear_3x(image)
bbb = average_image([aaa])
bbb.save("experiment b.png","PNG")
"""

#rrr = do_upscale(experiment_c(image),"experiment c")
#rrr2 = do_upscale(experiment_c(rrr),"experiment c 2")

"""
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
"""
