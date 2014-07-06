#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import argparse
import pprint
import sys
import unittest

import numpy

from nupic.research.TP import TP as TP



SHOW_ENABLED = False



# ==============================
# Tests
# ==============================

class TemporalPoolerBehaviorTest(unittest.TestCase):

  def testA(self):
    showTest("Basic first order sequences")
    tp = newTP()
    p = generatePatterns(tp.numberOfCols)

    showSegments(tp)
    sequence = [p[0], p[1], p[2], p[3]]
    feedTP(tp, sequence)

    sequence = [p[0], p[1], p[2]]
    feedTP(tp, sequence, reset=False, learn=False)
    self.assertEqual(len(getPredictions(tp)), 0)
    resetTP(tp)

    sequence = [p[0], p[1], p[2], p[3]]
    feedTP(tp, sequence, num=2)

    sequence = [p[0]]
    feedTP(tp, sequence, reset=False)
    self.assertEqual(len(getPredictions(tp)), 1)
    sequence = [p[1]]
    feedTP(tp, sequence, reset=False)
    sequence = [p[2]]
    feedTP(tp, sequence, reset=False)
    sequence = [p[3]]
    feedTP(tp, sequence, reset=False)
    resetTP(tp)

    sequence = [p[0], p[1], p[2], p[3]]
    feedTP(tp, sequence, num=3)

    sequence = [p[0]]
    feedTP(tp, sequence, reset=False, learn=False)
    sequence = [p[1]]
    feedTP(tp, sequence, reset=False, learn=False)
    sequence = [p[2]]
    feedTP(tp, sequence, reset=False, learn=False)
    resetTP(tp)

    sequence = [p[0], p[1], p[2], p[3]]
    feedTP(tp, sequence, num=5)

    sequence = [p[0], p[1], p[2]]
    feedTP(tp, sequence, reset=False, learn=False)
    self.assertEqual(len(getPredictions(tp)), 1)
    resetTP(tp)


  def testB(self):
    showTest("High order sequences")
    tp = newTP()
    p = generatePatterns(tp.numberOfCols)

    sequence = [p[0], p[1], p[2], p[3]]
    feedTP(tp, sequence, num=5)

    sequence = [p[1], p[2]]
    feedTP(tp, sequence, reset=False, learn=False)
    self.assertEqual(len(getPredictions(tp)), 1)
    resetTP(tp)

    sequence = [p[4], p[1], p[2], p[5]]
    feedTP(tp, sequence, num=5)

    sequence = [p[1], p[2]]
    feedTP(tp, sequence, reset=False, learn=False)
    self.assertEqual(len(getPredictions(tp)), 2)
    resetTP(tp)

    sequence = [p[0], p[1], p[2]]
    feedTP(tp, sequence, reset=False, learn=False)
    self.assertEqual(len(getPredictions(tp)), 1)
    resetTP(tp)


# ==============================
# TP
# ==============================

def newTP(overrides=None):
  params = {
    "numberOfCols": 6,
    "cellsPerColumn": 4,
    "initialPerm": 0.3,
    "connectedPerm": 0.5,
    "minThreshold": 1,
    "newSynapseCount": 2,
    "permanenceInc": 0.1,
    "permanenceDec": 0.05,
    "activationThreshold": 1,
    "globalDecay": 0,
    "burnIn": 1,
    "checkSynapseConsistency": False,
    "pamLength": 1,
    "maxInfBacktrack": 0,
    "maxLrnBacktrack": 0
  }
  params.update(overrides or {})
  tp = TP(**params)
  show("Initialized new TP with parameters:")
  show(pprint.pformat(params), newline=True)
  return tp


def feedTP(tp, sequence, learn=True, reset=True, num=1):
  showInput(sequence, learn=learn, reset=reset, num=num)

  for _ in range(num):
    for element in sequence:
      tp.compute(element, enableLearn=learn, computeInfOutput=True)
    if reset:
      tp.reset()

  if learn:
    showSegments(tp)

  if not reset:
    showActivations(tp)
    showPredictions(tp)

    noteText = "(connectedPerm: {0})".format(tp.connectedPerm)
    show(noteText, newline=True)


def resetTP(tp):
  tp.reset()
  showReset()


def formatTPState(state):
  values = []

  for i, col in enumerate(state):
    if sum(col) > 0:
      values.append([i, numpy.array(col, dtype=int)])

  return values


def getPredictions(tp):
  predictedState = tp.getPredictedState()
  return formatTPState(predictedState)


def getActivations(tp):
  activeState = tp.getActiveState()
  activeState = activeState.reshape([tp.numberOfCols, tp.cellsPerColumn])
  return formatTPState(activeState)


# ==============================
# Patterns
# ==============================

def generatePatterns(length):
  patterns = numpy.zeros([length, length], dtype=int)
  numpy.fill_diagonal(patterns, 1)
  showPatterns(patterns)
  return patterns


def getCodeForIndex(index):
  return chr(int(index) + 65)


def getCodeForPattern(pattern):
  return getCodeForIndex(pattern.nonzero()[0][0])


# ==============================
# Show
# ==============================

def show(text, newline=False):
  if SHOW_ENABLED:
    print text
    if newline:
      print


def showTest(text):
  show(("\n"
        "====================================\n"
        "Test: {0}\n"
        "===================================="
       ).format(text))


def showPatterns(patterns):
  show("Patterns: ")
  for pattern in patterns:
    show("{0}: {1}".format(getCodeForPattern(pattern), pattern))
  show("")


def showInput(sequence, reset=True, learn=True, num=1):
  sequenceText = ", ".join([getCodeForPattern(element) for element in sequence])
  resetText = ", reset" if reset else ""
  learnText = "(learning {0})".format("enabled" if learn else "disabled")
  numText = "[{0} times]".format(num) if num > 1 else ""
  show("Feeding sequence: {0}{1} {2} {3}".format(
       sequenceText, resetText, learnText, numText),
       newline=True)


def showSegments(tp):
  show("Segments: (format => [from column, from cell, permanence]) ")
  show("------------------------------------")
  for col in range(tp.numberOfCols):
    for cell in range(tp.cellsPerColumn):
      segments = []
      for seg in range(tp.getNumSegmentsInCell(col, cell)):
        synapses = tp.getSegmentOnCell(col, cell, seg)[1:]
        if len(synapses) == 0:
          continue
        segments.append(synapses)
      show("Col {0} ({1}) / Cell {2}: {3}".format(
        col, getCodeForIndex(col), cell, segments))
  show("------------------------------------", newline=True)


def showTPState(state, label):
  predictionsText = ", ".join(
    ["{0} (cells: {1})".format(
     getCodeForIndex(i),
     ", ".join([str(c) for c in cells.nonzero()[0].tolist()]))
       for (i, cells) in state])
  predictionsText = predictionsText or "None"

  show("{0} ({1}): {2}".format(
    label, len(state), predictionsText))


def showPredictions(tp):
  showTPState(getPredictions(tp), "Predictions")


def showActivations(tp):
  showTPState(getActivations(tp), "Activations")


def showReset():
  show("TP reset.", newline=True)


# ==============================
# Main
# ==============================

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--show', default=False, action='store_true')
  parser.add_argument('unittest_args', nargs='*')

  args = parser.parse_args()
  SHOW_ENABLED = args.show

  unitArgv = [sys.argv[0]] + args.unittest_args
  unittest.main(argv=unitArgv)
