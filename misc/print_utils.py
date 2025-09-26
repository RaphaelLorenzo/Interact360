#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
from print_color import print
from functools import wraps
import time
import datetime
import inspect
from millify import millify

def prSilent(text, *argv, logfile = None):
  text = str(text)
  for arg in argv:
    text += " "
    text += str(arg)
  if logfile is not None:
    log_in_file(text, logfile, "silent")

def prSuccess(text, *argv, logfile = None):
  text = str(text)
  for arg in argv:
    text += " "
    text += str(arg)
  print(text, tag = "ok", tag_color = "green", color = "white")
  if logfile is not None:
    log_in_file(text, logfile, "ok")
    
def prInfo(text, *argv, logfile = None):
  text = str(text)
  for arg in argv:
    text += " "
    text += str(arg)
  print(text, tag = "info", tag_color = "cyan", color = "white")
  if logfile is not None:
    log_in_file(text, logfile, "info")
    
def prInfoBold(text, *argv, logfile = None):
  text = str(text)
  for arg in argv:
    text += " "
    text += str(arg)
  print(text, tag = "info", tag_color = "cyan", color = "white", format = "bold")
  if logfile is not None:
    log_in_file(text, logfile, "info")
    
def prDebug(text, *argv, logfile = None):
  text = str(text)
  for arg in argv:
    text += " "
    text += str(arg)
  print(text, tag = "debug", tag_color = "red", background = "white", color = "white")
  if logfile is not None:
    log_in_file(text, logfile, "debug")
    
def prWarning(text, *argv, logfile = None):
  text = str(text)
  for arg in argv:
    text += " "
    text += str(arg)
  print(text, tag = "warning", tag_color = "yellow", color = "white")
  if logfile is not None:
    log_in_file(text, logfile, "warning")
    
def prError(text, *argv, logfile = None,):
  text = str(text)
  for arg in argv:
    text += " "
    text += str(arg)
  print(text, tag = "error", tag_color = "red", color = "white")
  if logfile is not None:
    log_in_file(text, logfile, "error")

def log_in_file(text, logfile, tag = ""):
  with open(logfile, "a") as f:
    now = datetime.datetime.now()
    ts = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
    stack = inspect.stack()
    caller = stack[2] # never use the function directly because 2 means there is parent (ie prXXX before calling log_in_file)
    logtext = "[{}] [{} | {}:{}] {}\n".format(tag, ts, caller.filename, caller.lineno, text)
    f.write(logtext)
            
def prTimer(text, tic, tac, logfile = None, silent = False):
  newtext = "{} {:.1f} ms".format(text, (tac-tic)*1000)
  if not silent:
    print(newtext, tag = "timer", tag_color = "purple", color = "white")
  if logfile is not None:
    log_in_file(newtext, logfile, "timer")
    
def timeit(func): 
    @wraps(func)
    def wrapper_function(*args, **kwargs): 
        tic = time.time()
        res = func(*args,  **kwargs) 
        tac = time.time()
        print("{} {:.1f} ms".format(func.__name__, (tac-tic)*1000), tag = "timer", tag_color = "purple", color = "white")
        return res
    return wrapper_function 

