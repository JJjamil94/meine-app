# meine-app
import SwiftUI
import AVFoundation
import Speech
import Foundation

// Global chat message model so OpenAIClient and ChatView share the same type
struct ChatMessage: Identifiable { let id = UUID(); let role: String; let text: String }

// MARK: - OpenAI minimal client (Chat Completions API)
struct OpenAIClient {
    let apiKey: String

    func reply(system: String, history: [ChatMessage]) async throws -> String {
        struct Req: Encodable {
            let model = "gpt-4o-mini"
            let messages: [PayloadMsg]
            struct PayloadMsg: Encodable { let role: String; let content: String }
        }
        struct Resp: Decodable {
            struct Choice: Decodable { struct Msg: Decodable { let content: String? }; let message: Msg }
            let choices: [Choice]
        }

        var msgs: [Req.PayloadMsg] = [ .init(role: "system", content: system) ]
        msgs += history.map { Req.PayloadMsg(role: $0.role, content: $0.text) }

        var r = URLRequest(url: URL(string: "https://api.openai.com/v1/chat/completions")!)
        r.httpMethod = "POST"
        r.addValue("application/json", forHTTPHeaderField: "Content-Type")
        r.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        r.httpBody = try JSONEncoder().encode( Req(messages: msgs) )

        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        let session = URLSession(configuration: config)

        let (data, resp) = try await session.data(for: r)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            let status = (resp as? HTTPURLResponse)?.statusCode ?? -1
            let body = String(data: data, encoding: .utf8) ?? ""
            throw NSError(domain: "OpenAI", code: status,
                          userInfo: [NSLocalizedDescriptionKey: "HTTP \(status): \(body)"])
        }
        let decoded = try JSONDecoder().decode(Resp.self, from: data)
        return decoded.choices.first?.message.content?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }
}

// MARK: - Utility functions for answer checking
func normalize(_ s: String) -> String {
    let lowered = s.folding(options: .diacriticInsensitive, locale: .current).lowercased()
    let allowed = CharacterSet.alphanumerics.union(.whitespaces)
    let punctFree = lowered.unicodeScalars.filter { allowed.contains($0) }.map(String.init).joined()
    let collapsed = punctFree.replacingOccurrences(
        of: "\\s+",
        with: " ",
        options: .regularExpression,
        range: punctFree.startIndex..<punctFree.endIndex
    )
    return collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
}

func levenshtein(_ a: String, _ b: String) -> Int {
    let A = Array(a), B = Array(b); let n=A.count, m=B.count
    if n==0 { return m }; if m==0 { return n }
    var dp = Array(0...m)
    for i in 1...n {
        var prev = dp[0]; dp[0] = i
        for j in 1...m {
            let t = dp[j]; let cost = (A[i-1]==B[j-1]) ? 0 : 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost); prev = t
        }
    }
    return dp[m]
}

func isCloseEnough(_ u: String, to c: String) -> Bool {
    let U = normalize(u), C = normalize(c); if U == C { return true }
    let dist = levenshtein(U, C)
    let tol = max(1, Int(Double(C.count) * 0.15))
    return dist <= tol
}

// MARK: - Speech Manager (unchanged logic)
final class SpeechManager: NSObject, ObservableObject {
    @Published var transcript = ""
    @Published var isAuthorized = false
    @Published var isRecording = false

    private let audioEngine = AVAudioEngine()
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    override init() { super.init(); requestAuth() }
    func requestAuth() {
        AVAudioApplication.requestRecordPermission { _ in }
        SFSpeechRecognizer.requestAuthorization { s in
            DispatchQueue.main.async { self.isAuthorized = (s == .authorized) }
        }
    }
    func start(locale: Locale) {
        guard !isRecording else { return }
        recognizer = SFSpeechRecognizer(locale: locale)
        transcript = ""
        request = SFSpeechAudioBufferRecognitionRequest()
        request?.shouldReportPartialResults = true
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.record, mode: .measurement, options: .duckOthers)
        try? session.setActive(true, options: .notifyOthersOnDeactivation)
        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buf, _ in
            self.request?.append(buf)
        }
        audioEngine.prepare(); try? audioEngine.start()
        task = recognizer?.recognitionTask(with: request!) { res, err in
            if let r = res { DispatchQueue.main.async { self.transcript = r.bestTranscription.formattedString } }
            if err != nil || (res?.isFinal ?? false) { self.stop() }
        }
        isRecording = true
    }
    func stop() {
        guard isRecording else { return }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        request?.endAudio()
        task?.cancel()
        isRecording = false
    }
}


// -------- Datenmodell & Sätze --------
struct Sentence: Identifiable, Codable, Hashable {
    let id: UUID
    let en: String
    let pt: String
    init(id: UUID = UUID(), en: String, pt: String) { self.id = id; self.en = en; self.pt = pt }
}

enum StudyPlan: String, CaseIterable, Identifiable {
    case daily, weekly, monthly
    var id: String { rawValue }
    var title: String { self == .daily ? "Daily Plan" : (self == .weekly ? "Weekly Plan" : "Monthly Plan") }
    var dailyTarget: Int { self == .daily ? 3 : (self == .weekly ? 10 : 20) }
}

enum AnswerMode: String, CaseIterable, Identifiable { case typing, speaking; var id: String { rawValue }; var title: String { self == .typing ? "Type" : "Speak" } }

let SENTENCES: [Sentence] = [
    Sentence(en:"How are you?", pt:"Como você está?"),
    Sentence(en:"I am hungry.", pt:"Estou com fome."),
    Sentence(en:"The weather is nice.", pt:"O tempo está bom."),
    Sentence(en:"I love you.", pt:"Eu te amo."),
    Sentence(en:"Where is the bathroom?", pt:"Onde fica o banheiro?"),
    Sentence(en:"I would like a coffee, please.", pt:"Eu gostaria de um café, por favor."),
    Sentence(en:"What time is it?", pt:"Que horas são?"),
    Sentence(en:"Can you help me?", pt:"Você pode me ajudar?"),
    Sentence(en:"I don't understand.", pt:"Eu não entendo."),
    Sentence(en:"I am learning Portuguese.", pt:"Eu estou aprendendo português."),
    Sentence(en:"See you tomorrow.", pt:"Até amanhã."),
    Sentence(en:"Where are you from?", pt:"De onde você é?"),
    Sentence(en:"How old are you?", pt:"Quantos anos você tem?"),
    Sentence(en:"I want a coffee.", pt:"Eu quero um café.")
]

final class ProgressStore {
    static let shared = ProgressStore()
    private let learnedKey = "learned_sentence_ids_v4"
    private let streakKey = "streak_count_v1"
    private let lastDoneKey = "streak_last_done_v1"
    var learnedIDs: Set<UUID> {
        get { Set((UserDefaults.standard.array(forKey: learnedKey) as? [String] ?? []).compactMap(UUID.init(uuidString:))) }
        set { UserDefaults.standard.set(Array(newValue).map{ $0.uuidString }, forKey: learnedKey) }
    }
    var streakCount: Int { get { UserDefaults.standard.integer(forKey: streakKey) } set { UserDefaults.standard.set(newValue, forKey: streakKey) } }
    var lastDoneDate: String { get { UserDefaults.standard.string(forKey: lastDoneKey) ?? "" } set { UserDefaults.standard.set(newValue, forKey: lastDoneKey) } }
    static func yyyyMMdd(_ d: Date) -> String { let f = DateFormatter(); f.calendar = .init(identifier:.gregorian); f.dateFormat = "yyyy-MM-dd"; return f.string(from: d) }
    func bumpStreakIfNeeded() {
        let today = Self.yyyyMMdd(Date()); if lastDoneDate == today { return }
        if let y = Calendar.current.date(byAdding: .day, value: -1, to: Date()), Self.yyyyMMdd(y) == lastDoneDate { streakCount += 1 } else { streakCount = 1 }
        lastDoneDate = today
    }
}

// -------- Learn Tab --------
struct LearnHomeView: View {
    @State private var selectedPlan: StudyPlan? = nil
    var body: some View {
        NavigationStack {
            if let plan = selectedPlan {
                StudyView(plan: plan) { selectedPlan = nil }
            } else {
                VStack(spacing: 16) {
                    ForEach(StudyPlan.allCases) { plan in
                        Button { selectedPlan = plan } label: {
                            HStack {
                                Text(plan.title).font(.headline)
                                Spacer()
                                Text("\(plan.dailyTarget) sentences/day").foregroundStyle(.secondary)
                            }
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(RoundedRectangle(cornerRadius: 12).fill(Color.gray.opacity(0.12)))
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal)
                    }
                    Spacer()
                }
                .navigationTitle("Choose your plan")
            }
        }
    }
}

// -------- StudyView --------
struct StudyView: View {
    let plan: StudyPlan
    var onBack: () -> Void

    @State private var sessionSet: [Sentence] = []
    @State private var current: Sentence?
    @State private var correctThisSession: Set<UUID> = []
    @State private var direction: String = "EN → PT"
    @State private var mode: AnswerMode = .typing
    @State private var userText: String = ""
    @State private var feedback: String = ""
    @StateObject private var speech = SpeechManager()

    private var target: Int { plan.dailyTarget }
    private var doneCount: Int { correctThisSession.count }
    private var isFinished: Bool { doneCount >= target }

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Button("Back") { onBack() }
                Spacer()
                Text(plan.title).font(.title3).bold()
                Spacer()
                Text("\(doneCount)/\(target)")
                    .padding(6)
                    .background(Capsule().fill(Color.gray.opacity(0.15)))
            }.padding(.horizontal)

            HStack {
                Picker("Direction", selection: $direction) {
                    Text("EN → PT").tag("EN → PT")
                    Text("PT → EN").tag("PT → EN")
                }.pickerStyle(.segmented)
                Picker("Mode", selection: $mode) {
                    ForEach(AnswerMode.allCases) { Text($0.title).tag($0) }
                }.pickerStyle(.segmented)
            }.padding(.horizontal)

            if isFinished {
                SuccessView(planTitle: plan.title, streak: ProgressStore.shared.streakCount) { startSession() }
                    .padding(.horizontal)
            } else if let cur = current {
                Text(direction == "EN → PT" ? cur.en : cur.pt)
                    .font(.title3)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                if mode == .typing {
                    TextField("Type your translation", text: $userText)
                        .textFieldStyle(.roundedBorder)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)
                        .padding(.horizontal)
                        .onSubmit { checkAnswer() }

                    Button(action: checkAnswer) {
                        Text("Check answer")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }.padding(.horizontal)
                } else {
                    VStack(spacing: 8) {
                        HStack {
                            if speech.isRecording {
                                Circle().frame(width:10,height:10).foregroundColor(.red)
                                Text("Recording…")
                            } else {
                                Text("Tap “Start recording” and speak your answer.")
                            }
                            Spacer()
                        }
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                        TextEditor(text: Binding(get:{ speech.transcript }, set:{ speech.transcript = $0 }))
                            .frame(minHeight:80,maxHeight:120)
                            .overlay(RoundedRectangle(cornerRadius:8).stroke(Color.gray.opacity(0.3)))
                            .padding(.horizontal)

                        HStack {
                            Button {
                                if speech.isRecording { speech.stop() }
                                else {
                                    let loc = (direction == "EN → PT") ? Locale(identifier:"pt-BR") : Locale(identifier:"en-US")
                                    speech.start(locale: loc)
                                }
                            } label: {
                                Text(speech.isRecording ? "Stop recording" : "Start recording")
                                    .frame(maxWidth:.infinity)
                                    .padding(.vertical,10)
                                    .background(speech.isRecording ? Color.red : Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                            Button {
                                userText = speech.transcript
                                speech.stop()
                                checkAnswer()
                            } label: {
                                Text("Check answer")
                                    .frame(maxWidth:.infinity)
                                    .padding(.vertical,10)
                                    .background(Color.green)
                                    .foregroundColor(.white)
                                    .cornerRadius(10)
                            }
                        }
                        .padding(.horizontal)
                    }
                }

                Text(feedback)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
            } else {
                Text("Loading…")
                    .foregroundStyle(.secondary)
                    .onAppear { startSession() }
            }

            Spacer()
        }
        .navigationBarBackButtonHidden(true)
        .onAppear { startSession() }
        .onChange(of: speech.isRecording) { _, newValue in
            if newValue == false, mode == .speaking,
               !speech.transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                userText = speech.transcript
                checkAnswer()
            }
        }
    }

    func startSession() {
        speech.stop(); speech.transcript = ""; feedback = ""; userText = ""; correctThisSession = []
        switch plan {
        case .daily:
            let wanted = ["Where are you from?","How old are you?","I want a coffee."]
            sessionSet = wanted.compactMap { w in SENTENCES.first { $0.en == w } }
            if sessionSet.count < 3 { sessionSet = Array(SENTENCES.prefix(3)) }
        case .weekly:
            sessionSet = Array(SENTENCES.prefix(min(10, SENTENCES.count)))
        case .monthly:
            sessionSet = Array(SENTENCES.prefix(min(20, SENTENCES.count)))
        }
        current = pickNextFromSession()
    }
    func pickNextFromSession() -> Sentence? {
        sessionSet.filter { !correctThisSession.contains($0.id) }.randomElement()
    }
    func checkAnswer() {
        guard let cur = current else { return }
        let correct = (direction == "EN → PT") ? cur.pt : cur.en
        let cand = (mode == .typing) ? userText : speech.transcript
        if isCloseEnough(cand, to: correct) {
            feedback = "✅ Correct!"
            correctThisSession.insert(cur.id)
            var learned = ProgressStore.shared.learnedIDs
            learned.insert(cur.id)
            ProgressStore.shared.learnedIDs = learned
            userText = ""; speech.transcript = ""
            if let next = pickNextFromSession() {
                current = next
            } else {
                current = nil
                ProgressStore.shared.bumpStreakIfNeeded()
            }
        } else {
            feedback = "❌ Incorrect. Correct answer: \(correct)"
        }
    }
}

// -------- Erfolgsscreen --------
struct SuccessView: View {
    let planTitle: String
    let streak: Int
    var onRepeat: () -> Void
    @State private var showShare = false
    var body: some View {
        VStack(spacing: 16) {
            Text("🎉 Hooray!").font(.largeTitle).bold()
            Text("\(planTitle) completed successfully.").font(.title3)
            Text("Great work! Keep going — repetition makes it stick.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .padding(.horizontal)
            Text("🔥 Streak: \(streak) days in a row")
                .padding(8)
                .background(Capsule().fill(Color.orange.opacity(0.15)))
            HStack {
                Button { onRepeat() } label: {
                    Text("Practice again")
                        .frame(maxWidth:.infinity)
                        .padding(.vertical,12)
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
                Button { showShare = true } label: {
                    Text("Share")
                        .frame(maxWidth:.infinity)
                        .padding(.vertical,12)
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
                .sheet(isPresented: $showShare) {
                    ShareSheet(items:["I just finished my \(planTitle)! 🔥 Streak: \(streak) days."])
                }
            }
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

// -------- Chat Tab --------
struct ChatView: View {
    private let ai = OpenAIClient(apiKey: "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")
    @StateObject private var speech = SpeechManager()
    private let tts = AVSpeechSynthesizer()

    @State private var messages: [ChatMessage] = [
        .init(role: "assistant", text: "Hi! Let’s practice English ↔︎ Portuguese. Speak or type a message.")
    ]
    @State private var inputText = ""
    @State private var busy = false
    @State private var errorText: String?

    var body: some View {
        VStack {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    ForEach(messages) { m in
                        HStack {
                            if m.role == "assistant" {
                                Text(m.text)
                                    .padding(10)
                                    .background(Color.gray.opacity(0.12))
                                    .cornerRadius(10)
                                Spacer()
                            } else {
                                Spacer()
                                Text(m.text)
                                    .padding(10)
                                    .background(Color.blue.opacity(0.15))
                                    .cornerRadius(10)
                            }
                        }
                    }
                }.padding()
            }

            if let err = errorText {
                Text(err).foregroundColor(.red).font(.footnote).padding(.horizontal)
            }

            HStack(spacing: 8) {
                Button {
                    if speech.isRecording { speech.stop(); inputText = speech.transcript }
                    else { speech.start(locale: Locale(identifier: "en-US")) }
                } label: {
                    Image(systemName: speech.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                        .font(.system(size: 28))
                }

                TextField("Type here…", text: $inputText)
                    .textFieldStyle(.roundedBorder)

                Button { Task { await send() } } label: {
                    if busy { ProgressView() } else { Image(systemName: "paperplane.fill") }
                }
                .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || busy)
            }
            .padding()
        }
        .navigationTitle("AI Chat")
    }

    func send() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(.init(role: "user", text: text))
        inputText = ""; errorText = nil; busy = true
        do {
            let reply = try await ai.reply(
                system: "You are a friendly bilingual tutor (English and Portuguese). Keep replies short and speak clearly.",
                history: messages
            )
            messages.append(.init(role: "assistant", text: reply))
            let utter = AVSpeechUtterance(string: reply)
            utter.voice = AVSpeechSynthesisVoice(language: "en-US")
            tts.speak(utter)
        } catch {
            errorText = "AI error: \(error.localizedDescription)"
        }
        busy = false
    }
}

// -------- Root Tabs --------
struct ContentView: View {
    var body: some View {
        TabView {
            LearnHomeView()
                .tabItem { Label("Learn", systemImage: "book.fill") }
            ChatView()
                .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right.fill") }
        }
    }
}

