from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


class YTVideoRAGApplication:
    def __init__(self):
        self.video_id = 'cp0iOk7TiXU'
        self.llm = ChatOllama(model = 'gemma2:2b-instruct-q5_0', temperature=0)
        self.embeddings = OllamaEmbeddings(model='nomic-embed-text')
        self.query = '''
        how to make the powerpoint slides
        '''

    def get_video_transcript(self, preserve_formatting=None):
        '''
        Method to extract the transcript of the youtube video
        '''
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(self.video_id, languages=['en'])
            self.transcript = ''.join(chunk.text for chunk in transcript_list)
        except TranscriptsDisabled:
            print('No caption available for this video')

    def splitting_transcript(self):
        '''
        Method to split the transcript into chunks
        :return: Chunks of document
        '''
        try:
            self.get_video_transcript()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
            )
            self.chunks = splitter.create_documents([self.transcript])
        except Exception as e:
            print(f"Error while splitting transcript: {e}")

    def create_embedding(self):
        '''
        Method to create the vector store for embedding
        :return: vector store
        '''
        try:
            self.splitting_transcript()
            self.vector_store = FAISS.from_documents(documents=self.chunks, embedding=self.embeddings)
        except Exception as e:
            print(f"Error while creating embedding: {e}")

    def create_retriever(self):
        '''
        Method to create the retriever
        :return: retriever
        '''
        self.create_embedding()
        try:
            self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={'k':3})
        except Exception as e:
            print(f"Error while creating retriever: {e}")

    def creating_prompt(self):
        '''
        Method to create the prompt
        :return: final prompt
        '''
        self.create_retriever()
        prompt = PromptTemplate(
            template='''
            You are a helpful assistant.
            Answer only from the provided transcript context.
            If the context is insufficient, just say you don't know.
            {context}
            Question: {question}
            ''',
            input_variables=['context', 'question'],
        )
        retriever_docs = self.retriever.invoke(self.query)
        context_text = '\n\n'.join(doc.page_content for doc in retriever_docs)
        final_prompt= prompt.invoke({"context": context_text, "question": self.query})

        ans = self.llm.invoke(final_prompt)
        print(ans.content)

if __name__ == '__main__':
    yt_app = YTVideoRAGApplication()
    yt_app.creating_prompt()

Response = '''
This tutorial shows how to create a welcome slide in Microsoft PowerPoint. 
Here's a breakdown of the steps:

1. **Create a Blank Slide:** Start with a blank slide in PowerPoint.
2. **Add a Rectangle Shape:**  
   - Select "Insert" > "Shape" > "Rectangle"
   - Drag the rectangle to cover the entire slide.
3. **Set the Shape Outline:**
   - Go to "Shape Outline" > "No Outline"
4. **Choose a Fill Color:**
   - Select "Shape Fill" > "Gradient"
   - Choose a gradient from the center.
5. **Add Text:**
   - Select the rectangle and text box.
   - Change the font to bold.
   - Increase the font size.
   - Place the text where you want it.
6. **Add Pictures:**
   - Select "Insert" > "Picture"
   - Choose a picture.
   - Adjust the aspect ratio using "Crop" and place it.
   - Send the picture to the back.
7. **Add More Text:**
   - Select "Insert" > "Shape" > "Text Box"
   - Drag it to the desired location.
   - Write "Welcome" in the text box.
   - Change the font, size, and spacing. 
   - Align the text box to the center. 
 
Let me know if you'd like more details on any specific step! 
'''

