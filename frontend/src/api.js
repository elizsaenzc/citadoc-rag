async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, options)
  if (!response.ok) {
    let detail = `Error ${response.status}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail)
  }
  return response.status === 204 ? null : response.json()
}

const json = body => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body)
})

export const api = {
  health: () => request('/health'),
  documents: () => request('/documents'),
  upload: file => {
    const form = new FormData()
    form.append('file', file)
    return request('/documents', { method: 'POST', body: form })
  },
  remove: id => request(`/documents/${id}`, { method: 'DELETE' }),
  ask: (question, docIds, conversationId) =>
    request('/ask', json({ question, doc_ids: docIds?.length ? docIds : null, conversation_id: conversationId })),
  agent: (question, conversationId) => request('/agent', json({ question, conversation_id: conversationId })),
  conversations: () => request('/conversations'),
  conversation: id => request(`/conversations/${id}`),
  removeConversation: id => request(`/conversations/${id}`, { method: 'DELETE' })
}
