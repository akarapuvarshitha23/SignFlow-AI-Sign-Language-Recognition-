/**
 * SignFlow AI - Comprehensive Sign Language Dictionary & SVG Illustrations
 * Covers Alphabets (A-Z), Numbers (0-9), and Common Communicative Phrases.
 */

const SIGN_DICTIONARY = [
    // --- Alphabets (A - Z) ---
    {
        char: "A",
        name: "Letter A",
        category: "alphabet",
        description: "Fist closed tightly with thumb resting upright along the index finger.",
        tips: "Ensure all four fingers are curled down flat; thumb points straight up along the side."
    },
    {
        char: "B",
        name: "Letter B",
        category: "alphabet",
        description: "Four fingers held straight up touching together; thumb tucked flat across the palm.",
        tips: "Keep fingers pressed together firmly with no gap between them."
    },
    {
        char: "C",
        name: "Letter C",
        category: "alphabet",
        description: "Hand curved like a cup or semicircle forming the letter 'C'.",
        tips: "Curve both fingers and thumb in a gentle arc facing sideways."
    },
    {
        char: "D",
        name: "Letter D",
        category: "alphabet",
        description: "Index finger points straight up; thumb touches the middle fingertip forming a circle.",
        tips: "Only the index finger is extended; ring and pinky fingers are tucked into the palm."
    },
    {
        char: "E",
        name: "Letter E",
        category: "alphabet",
        description: "All four fingers curled tightly downwards with tips resting on the thumb folded underneath.",
        tips: "Fingertips curl back towards palm, resting right on the thumb cushion."
    },
    {
        char: "F",
        name: "Letter F",
        category: "alphabet",
        description: "Index finger and thumb touch tips forming an 'OK' circle; remaining 3 fingers extended.",
        tips: "Spread middle, ring, and pinky fingers slightly upward."
    },
    {
        char: "G",
        name: "Letter G",
        category: "alphabet",
        description: "Index finger and thumb extend horizontally parallel pointing sideways like a pincher.",
        tips: "Index finger is parallel to thumb, pointing to the left/side."
    },
    {
        char: "H",
        name: "Letter H",
        category: "alphabet",
        description: "Index and middle fingers extended together horizontally pointing sideways.",
        tips: "Keep index and middle fingers touching each other."
    },
    {
        char: "I",
        name: "Letter I",
        category: "alphabet",
        description: "Pinky finger extended straight up; other three fingers curled into a fist held by thumb.",
        tips: "Only the little pinky finger stays upright."
    },
    {
        char: "J",
        name: "Letter J",
        category: "alphabet",
        description: "Form the letter 'I' with pinky, then draw a 'J' curve in the air.",
        tips: "Trace a small hook trajectory downwards and inward."
    },
    {
        char: "K",
        name: "Letter K",
        category: "alphabet",
        description: "Index points up; middle finger extends forward at an angle; thumb rests between them.",
        tips: "Thumb tip pushes against the middle finger knuckle."
    },
    {
        char: "L",
        name: "Letter L",
        category: "alphabet",
        description: "Thumb and index finger extended at a 90-degree right angle forming the letter 'L'.",
        tips: "Index points straight up, thumb extends straight left."
    },
    {
        char: "M",
        name: "Letter M",
        category: "alphabet",
        description: "Thumb tucked under the first three fingers (Index, Middle, Ring) curled into a fist.",
        tips: "Three finger ridges rest visibly over the thumb."
    },
    {
        char: "N",
        name: "Letter N",
        category: "alphabet",
        description: "Thumb tucked under the first two fingers (Index and Middle) curled into a fist.",
        tips: "Two finger ridges rest over the thumb."
    },
    {
        char: "O",
        name: "Letter O",
        category: "alphabet",
        description: "All five fingertips touch together forming an 'O' shaped circle.",
        tips: "Create an open circular ring with fingers and thumb tips meeting."
    },
    {
        char: "P",
        name: "Letter P",
        category: "alphabet",
        description: "Hand shaped like 'K' tilted downwards so index finger points horizontally.",
        tips: "Middle finger points down towards the ground."
    },
    {
        char: "Q",
        name: "Letter Q",
        category: "alphabet",
        description: "Hand shaped like 'G' tilted downwards towards the ground.",
        tips: "Index and thumb point down like grasping a coin."
    },
    {
        char: "R",
        name: "Letter R",
        category: "alphabet",
        description: "Index and middle fingers crossed over each other ('fingers crossed').",
        tips: "Middle finger curls over the back of the index finger."
    },
    {
        char: "S",
        name: "Letter S",
        category: "alphabet",
        description: "Fist with thumb wrapped tightly across the front of all curled fingers.",
        tips: "Unlike 'A' where thumb is alongside, in 'S' thumb crosses over the front knuckles."
    },
    {
        char: "T",
        name: "Letter T",
        category: "alphabet",
        description: "Thumb tucked under the index finger only; fist closed.",
        tips: "Thumb tip peeks out between index and middle knuckles."
    },
    {
        char: "U",
        name: "Letter U",
        category: "alphabet",
        description: "Index and middle fingers extended straight up touching together.",
        tips: "Keep the two fingers touching without spreading."
    },
    {
        char: "V",
        name: "Letter V",
        category: "alphabet",
        description: "Index and middle fingers extended straight up and spread apart in a 'V' shape (Peace).",
        tips: "Spread index and middle fingers apart forming an angle."
    },
    {
        char: "W",
        name: "Letter W",
        category: "alphabet",
        description: "Index, middle, and ring fingers extended and spread upright forming 'W'.",
        tips: "Pinky and thumb touch across the palm."
    },
    {
        char: "X",
        name: "Letter X",
        category: "alphabet",
        description: "Index finger hooked or bent like a small pirate hook; others curled into a fist.",
        tips: "Bend the top knuckle of the index finger forward."
    },
    {
        char: "Y",
        name: "Letter Y",
        category: "alphabet",
        description: "Thumb and pinky fingers extended outward; middle three curled (Hang Loose / Call Me).",
        tips: "Extend thumb and pinky as far apart as comfortable."
    },
    {
        char: "Z",
        name: "Letter Z",
        category: "alphabet",
        description: "Index finger extended, tracing the letter 'Z' in the air.",
        tips: "Trace horizontal, diagonal down, horizontal line."
    },

    // --- Numbers (0 - 9) ---
    {
        char: "0",
        name: "Number 0",
        category: "number",
        description: "Fingertips curved touching the thumb forming a zero / 'O' shape.",
        tips: "Similar to the letter O posture."
    },
    {
        char: "1",
        name: "Number 1",
        category: "number",
        description: "Index finger pointing straight up; thumb holds down middle, ring, pinky.",
        tips: "Palm faces towards you or forward."
    },
    {
        char: "2",
        name: "Number 2",
        category: "number",
        description: "Index and middle fingers extended straight up in a 'V' shape.",
        tips: "Palm facing towards your chest in traditional ASL counting."
    },
    {
        char: "3",
        name: "Number 3",
        category: "number",
        description: "Thumb, index, and middle fingers extended; ring and pinky curled down.",
        tips: "Notice in ASL, 3 uses the thumb, index, and middle fingers."
    },
    {
        char: "4",
        name: "Number 4",
        category: "number",
        description: "Four fingers extended straight up and spread; thumb folded across palm.",
        tips: "All four fingers up; thumb stays tucked in."
    },
    {
        char: "5",
        name: "Number 5",
        category: "number",
        description: "Open palm with all five fingers extended and spread wide.",
        tips: "Spread all 5 fingers naturally."
    },
    {
        char: "6",
        name: "Number 6",
        category: "number",
        description: "Thumb touches tip of pinky finger; index, middle, and ring fingers extended up.",
        tips: "The pinky is the smallest finger, representing 6."
    },
    {
        char: "7",
        name: "Number 7",
        category: "number",
        description: "Thumb touches tip of ring finger; index, middle, and pinky extended up.",
        tips: "Ring finger meets the thumb pad."
    },
    {
        char: "8",
        name: "Number 8",
        category: "number",
        description: "Thumb touches tip of middle finger; index, ring, and pinky extended up.",
        tips: "Middle finger meets the thumb pad."
    },
    {
        char: "9",
        name: "Number 9",
        category: "number",
        description: "Thumb touches tip of index finger (like letter F); remaining 3 fingers extended.",
        tips: "Index finger meets the thumb pad."
    },

    // --- Communicative Gestures & Phrases ---
    {
        char: "HELLO",
        name: "Hello / Greeting",
        category: "phrase",
        description: "Open flat hand starts near forehead/temple and moves outwards in a gentle salute/wave.",
        tips: "A universally recognized, friendly greeting gesture."
    },
    {
        char: "THANK_YOU",
        name: "Thank You",
        category: "phrase",
        description: "Fingertips of flat hand touch chin or lips and move forward towards the person.",
        tips: "Smile and gesture forward with sincerity."
    },
    {
        char: "YES",
        name: "Yes (Affirmative)",
        category: "phrase",
        description: "Fist held in front of chest, nodding up and down from the wrist like a head nodding.",
        tips: "Nod twice gently from the wrist."
    },
    {
        char: "NO",
        name: "No (Negative)",
        category: "phrase",
        description: "Index and middle fingers snap together against the thumb tip like a beak closing.",
        tips: "Quick, crisp snapping motion."
    },
    {
        char: "I_LOVE_YOU",
        name: "I Love You (ILY)",
        category: "phrase",
        description: "Thumb, index finger, and pinky finger all extended outwards; middle and ring curled down.",
        tips: "Combines letters 'I', 'L', and 'Y' into the worldwide sign for love."
    }
];

/**
 * Generates an aesthetic stylized hand silhouette SVG vector for any sign.
 */
function getSignSvg(signChar) {
    // Elegant stylized vector illustrations with soft lavender & sky blue accents
    return `
    <svg viewBox="0 0 100 100" class="sign-svg-illustration" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="handGrad_${signChar}" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#8b5cf6" />
                <stop offset="50%" stop-color="#6366f1" />
                <stop offset="100%" stop-color="#38bdf8" />
            </linearGradient>
            <filter id="glow_${signChar}">
                <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#8b5cf6" flood-opacity="0.25"/>
            </filter>
        </defs>
        
        <!-- Background Soft Circle -->
        <circle cx="50" cy="50" r="44" fill="#f8fafc" stroke="#e0e7ff" stroke-width="2"/>
        
        <!-- Hand Base / Palm Silhouette Representation -->
        <path d="M 32 75 C 32 60, 36 50, 42 45 C 46 42, 54 42, 58 45 C 64 50, 68 60, 68 75 Z" 
              fill="url(#handGrad_${signChar})" opacity="0.15" />
        
        <!-- Dynamic Sign Center Symbol / Posture Vector -->
        <circle cx="50" cy="46" r="24" fill="url(#handGrad_${signChar})" filter="url(#glow_${signChar})"/>
        
        <!-- Big Clear Letter / Digit Display -->
        <text x="50" y="55" font-family="'Plus Jakarta Sans', sans-serif" font-size="22" font-weight="800" 
              fill="#ffffff" text-anchor="middle" dominant-baseline="central">
            ${signChar.replace('_', ' ')}
        </text>
        
        <!-- Aesthetic Hand Pose Accent Lines -->
        <circle cx="50" cy="18" r="4" fill="#38bdf8" />
        <line x1="50" y1="22" x2="50" y2="28" stroke="#38bdf8" stroke-width="2" stroke-linecap="round"/>
    </svg>
    `;
}

