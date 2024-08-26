import sensenova

chat_completion = sensenova.ChatCompletion.create(
    model="SenseChat-32K",
    messages=[{"role": "user", "content": "Say this is a test!"}]
)

# print the chat completion
print(chat_completion.data.choices[0].message)

embedding = sensenova.Embedding.create(model="nova-embedding-stable",input=["What I Worked On\\n\\nFebruary 2021\\n\\nBefore college the two main things I worked on, outside of school, were writing and programming. I didn\'t write essays."])

print(embedding.embeddings[0].embedding)