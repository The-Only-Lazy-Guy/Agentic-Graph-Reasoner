def _contains_any_phrase(text, phrases):
    low = str(text or '').lower()
    return any(str(phrase or '').lower() in low for phrase in phrases)

q = "If astronauts can see sunlight in space, why can't they hear it there?"
q3 = "Can sound travel in the vacuum of space, or is it only light that propagates?"
low = q.lower()
low3 = q3.lower()

m_sound = _contains_any_phrase(low, ('sound', 'hear', 'hearing', 'audible', 'audio', 'acoustic', 'sonic', 'noise'))
m_light = _contains_any_phrase(low, ('light', 'sunlight', 'starlight', 'laser', 'flash', 'visible', 'see', 'seeing', 'sight', 'star', 'stars', 'sun'))
m_vac = _contains_any_phrase(low, ('vacuum', 'space', 'outer space', 'empty space', 'airless', 'without air', 'no air'))
print(f'Q2: {m_sound=} {m_light=} {m_vac=}')

m_sound = _contains_any_phrase(low3, ('sound', 'hear', 'hearing', 'audible', 'audio', 'acoustic', 'sonic', 'noise'))
m_light = _contains_any_phrase(low3, ('light', 'sunlight', 'starlight', 'laser', 'flash', 'visible', 'see', 'seeing', 'sight', 'star', 'stars', 'sun'))
m_vac = _contains_any_phrase(low3, ('vacuum', 'space', 'outer space', 'empty space', 'airless', 'without air', 'no air'))
print(f'Q3: {m_sound=} {m_light=} {m_vac=}')
