<template>
    <div class="page" v-loading="loading" element-loading-background="rgba(0, 0, 0, 0.45)">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
                </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="params.target = images[params.input]">
                    <el-option v-for="image in Object.keys(images)" :key="images[image]" :value='image'/>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class="value" type='number' size="small" v-model="params.target"  @change="update"/></div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="input"><img :src="params.input" crossorigin='anonymous'/></div>
            <div class="sliency">
                <div class="image-wrapper"><img class="image" :src="res.colorful" crossorigin='anonymous'/><div class="caption">Gradient</div></div>
                <div class="image-wrapper"><img class="image" :src="res.grayscale" crossorigin='anonymous'/><div class="caption">Saliency</div></div>
                <div class="image-wrapper"><img class="image" :src="res.grad_x_image" crossorigin='anonymous'/><div class="caption">Saliency * Image</div></div>
            </div>
            <div class="sliency">
                <div class="image-wrapper"><img class="image" :src="res.guided_colorful" crossorigin='anonymous'/><div class="caption">Guided Gradient</div></div>
                <div class="image-wrapper"><img class="image" :src="res.guided_grayscale" crossorigin='anonymous'/><div class="caption">Guided Saliency</div></div>
                <div class="image-wrapper"><img class="image" :src="res.guided_grad_x_image" crossorigin='anonymous'/><div class="caption">Guided Saliency * Image</div></div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'saliency',
    data() {
        return {
            models: [],
            images: {},
            activations: [],
            params: {
                model: 'alexnet',
                input: '',
                target: null
            },
            res: {},
            loading: false
        };
    },
    created() {
        this.config()
    },
    sockets: {
        response_saliecy(data) {
            this.res = data
            this.loading = false
        },

        models(data) {
            this.models = data
        },
        images(data) {
            this.images = data

            this.params.input = Object.keys(data)[0]
            this.params.target = this.images[this.params.input]
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.$socket.emit('get_images')
        },
        update() {
            this.$socket.emit("saliency", this.params);
            this.loading = true
        }
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        display: flex;
        flex-flow: row;

        align-items: center;
        justify-items: center;

        .input {
            flex: 0 0 auto;
        }

        .sliency {
            flex: 1 1 auto;
            display: flex;
            flex-flow: column;
            align-items: center;
            justify-items: center;
            img {
                padding: 10px;
            }
        }
    }
}

</style>
