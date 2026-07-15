export const API_BASE = 'http://localhost:8000';

export const SCALE_COLORS = {
  ok: '#10B981', noscan: '#94A3B8', unknown: '#FCD34D',
  err_1: '#9CA3AF', err_2: '#F59E0B', err_3: '#F97316',
  err_4: '#8B5CF6', err_7: '#EF4444', err_8: '#3B82F6', err_9: '#EC4899',
};

export const ODS_LABELS = {
  '1': 'Diverted',           '2': 'Dest. not reached',    '3': 'Dest. not available',
  '4': 'Failed to divert',   '5': 'Invalid destination',  '6': 'Unreachable dest.',
  '7': 'Dest. not received', '8': 'Condition invalid',    '9': 'Dest. not requested',
  'A': 'Flow restricted',    'B': 'Max. recirculation',   'C': 'Dest. override',
  '0': 'Unknown state',
};

export const ODS_COLORS = {
  '1': '#10B981', '2': '#EF4444', '3': '#F97316', '4': '#F59E0B',
  '5': '#6366F1', '6': '#EC4899', '7': '#8B5CF6', '8': '#0EA5E9',
  '9': '#14B8A6', 'A': '#84CC16', 'B': '#991B1B', 'C': '#64748B', '0': '#9CA3AF',
};

export const BLOCKED_FLAG_LABELS = {
  'FrontFault':              'Front Fault',
  'RearFault':               'Rear Fault',
  'MultipleCarriers':        'Multiple Carriers',
  'MultipleFault':           'Multiple Fault',
  'SmallItemOverlappingGap': 'Small Item / Gap',
  'Unsortable':              'Unsortable',
  'ScreenFault':             'Screen Fault',
  'SortRestricted':          'Sort Restricted',
  'MultipleDataForOneItem':  'Multiple Data / Item',
};

export const BLOCKED_FLAG_COLORS = {
  'FrontFault':              '#F59E0B',
  'RearFault':               '#F97316',
  'MultipleCarriers':        '#6366F1',
  'MultipleFault':           '#EF4444',
  'SmallItemOverlappingGap': '#14B8A6',
  'Unsortable':              '#EC4899',
  'ScreenFault':             '#8B5CF6',
  'SortRestricted':          '#0EA5E9',
  'MultipleDataForOneItem':  '#84CC16',
};
