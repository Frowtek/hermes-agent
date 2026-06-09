import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages } from '@/hermes'

import { exportSession } from './session-export'

vi.mock('@/hermes', () => ({
  getSessionMessages: vi.fn()
}))

const mockedGetSessionMessages = vi.mocked(getSessionMessages)

describe('exportSession', () => {
  beforeEach(() => {
    mockedGetSessionMessages.mockReset()
    mockedGetSessionMessages.mockResolvedValue({ messages: [] } as never)
    globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock')
    globalThis.URL.revokeObjectURL = vi.fn()
    vi.spyOn(HTMLElement.prototype, 'click').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // Regression: a session owned by another profile (Command Center's all-profile
  // list passes the whole session object) must read from that profile's backend,
  // not the active gateway profile's — otherwise the transcript 404s or comes
  // back wrong. getSessionMessages takes the owning profile to route the read.
  it('forwards the owning profile from the session object', async () => {
    await exportSession('s1', { session: { id: 's1', profile: 'work' } as never, title: 'Demo' })

    expect(mockedGetSessionMessages).toHaveBeenCalledWith('s1', 'work')
  })

  it('prefers an explicitly passed profile (sidebar action menu path)', async () => {
    await exportSession('s1', { profile: 'work', title: 'Demo' })

    expect(mockedGetSessionMessages).toHaveBeenCalledWith('s1', 'work')
  })

  it('omits the profile for current/default-profile sessions', async () => {
    await exportSession('s1', { title: 'Demo' })

    expect(mockedGetSessionMessages).toHaveBeenCalledWith('s1', undefined)
  })
})
